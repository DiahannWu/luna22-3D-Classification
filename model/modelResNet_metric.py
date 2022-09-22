import torch
import torch.nn.functional as F
from networks.ResNet2d import ResNet2d
from networks.ResNet3d import ResNet3d
from networks.VGG3d import VGG3d
from model.dataset import datasetModelClassifywithopencv, datasetModelClassifywithnpy
from torch.utils.data import DataLoader
from model.losses import BinaryFocalLoss, BinaryCrossEntropyLoss, MutilFocalLoss, MutilCrossEntropyLoss
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from model.metric import calc_accuracy
from model.visualization import plot_result
from pathlib import Path
import time
import os
import cv2
import multiprocessing
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

class BinaryResNet2dModel(object):
    """
    ResNet2d with binary class,should rewrite the dataset class and inference fucntion
    """

    def __init__(self, image_height, image_width, image_channel, numclass, batch_size,
                 loss_name='BinaryCrossEntropyLoss', inference=False, model_path=None, use_cuda=True):
        self.batch_size = batch_size
        self.loss_name = loss_name
        self.accuracyname = 'accu'
        self.image_height = image_height
        self.image_width = image_width
        self.image_channel = image_channel
        self.numclass = numclass

        self.alpha = 0.25
        self.gamma = 2

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = ResNet2d(self.image_channel, self.numclass)
        self.model.to(device=self.device)

        if inference:
            print(f'Loading model {model_path}')
            print(f'Using device {self.device}')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded!')

    def _dataloder(self, images, labels, shuffle=False):
        """"
        train dataset shuffle is true,validation is false
        """
        # Number of workers
        dataset = datasetModelClassifywithopencv(images, labels,
                                                 targetsize=(self.image_channel, self.image_height, self.image_width))
        # fow window num_workers is only zero,for linux num_workers can not zero
        # num_cpu = multiprocessing.cpu_count()
        num_cpu = 0
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=self.batch_size, num_workers=num_cpu,
                                pin_memory=True)
        return dataloader

    def _loss_function(self, lossname):
        if lossname is 'BinaryCrossEntropyLoss':
            return BinaryCrossEntropyLoss()
        if lossname is 'BinaryFocalLoss':
            return BinaryFocalLoss(alpha=self.alpha, gamma=self.gamma)

    def _accuracy_function(self, accuracyname, input, target):
        if accuracyname is 'accu':
            if self.numclass == 1:
                input = (input > 0.5).float()
                target = (target > 0.5).float()
                return calc_accuracy(input, target)
            else:
                input = torch.argmax(input, 1)
                return calc_accuracy(input, target)

    def trainprocess(self, trainimage, trainmask, validationimage, validationmask, model_dir, epochs=50, lr=1e-3):
        print("[INFO] training the network...")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        MODEL_PATH = os.path.join(model_dir, "BinaryResNet2d.pth")
        summary(self.model, input_size=(self.image_channel, self.image_height, self.image_width))
        print(self.model)
        # 1、initialize loss function and optimizer
        lossFunc = self._loss_function(self.loss_name)
        opt = optim.Adam(self.model.parameters(), lr=lr)
        # 2、load data train and validation dataset
        train_loader = self._dataloder(trainimage, trainmask, True)
        val_loader = self._dataloder(validationimage, validationmask)
        # 3、initialize a dictionary to store training history
        H = {"train_loss": [], "train_accuracy": [], "valdation_loss": [], "valdation_accuracy": []}
        # 4、start loop training wiht epochs times
        startTime = time.time()
        best_validation_dsc = 0.0
        # Tensorboard summary
        writer = SummaryWriter(log_dir=model_dir)
        for e in tqdm(range(epochs)):
            # 4.1、set the model in training mode
            self.model.train()
            # 4.2、initialize the total training and validation loss
            totalTrainLoss = []
            totalTrainAccu = []
            totalValidationLoss = []
            totalValiadtionAccu = []
            # 4.3、loop over the training set
            for batch in train_loader:
                # x should tensor with shape (N,C,W,H)
                x = batch['image']
                # y should tensor with shape (N,C,W,H),
                # if have mutil label y should one-hot,if only one label,the C is one
                y = batch['label']
                # send the input to the device
                x, y = x.to(self.device), y.to(self.device)
                # perform a forward pass and calculate the training loss and accu
                pred_logit = self.model(x)
                loss = lossFunc(pred_logit, y)
                pred = F.sigmoid(pred_logit)
                accu = self._accuracy_function(self.accuracyname, pred, y)
                # first, zero out any previously accumulated gradients,
                # then perform backpropagation,
                # and then update model parameters
                opt.zero_grad()
                loss.backward()
                opt.step()
                # add the loss to the total training loss so far
                totalTrainLoss.append(loss)
                totalTrainAccu.append(accu)
            # 4.4、switch off autograd and loop over the validation set
            with torch.no_grad():
                # set the model in evaluation mode
                self.model.eval()
                # loop over the validation set
                for batch in val_loader:
                    # x should tensor with shape (N,C,W,H)
                    x = batch['image']
                    # y should tensor with shape (N,C,W,H)
                    y = batch['label']
                    # send the input to the device
                    (x, y) = (x.to(self.device), y.to(self.device))
                    # make the predictions and calculate the validation loss
                    pred_logit = self.model(x)
                    loss = lossFunc(pred_logit, y)
                    pred = F.sigmoid(pred_logit)
                    accu = self._accuracy_function(self.accuracyname, pred, y)
                    totalValidationLoss.append(loss)
                    totalValiadtionAccu.append(accu)
            # 4.5、calculate the average training and validation loss
            avgTrainLoss = torch.mean(torch.stack(totalTrainLoss))
            avgValidationLoss = torch.mean(torch.stack(totalValidationLoss))
            avgTrainAccu = torch.mean(torch.stack(totalTrainAccu))
            avgValidationAccu = torch.mean(torch.stack(totalValiadtionAccu))
            # 4.6、update our training history
            H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            H["valdation_loss"].append(avgValidationLoss.cpu().detach().numpy())
            H["train_accuracy"].append(avgTrainAccu.cpu().detach().numpy())
            H["valdation_accuracy"].append(avgValidationAccu.cpu().detach().numpy())
            # 4.7、print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
            print("Train loss: {:.5f}, Train accu: {:.5f}，validation loss: {:.5f}, validation loss: {:.5f}".format(
                avgTrainLoss, avgTrainAccu, avgValidationLoss, avgValidationAccu))
            # Record training loss and accuracy for each phase
            writer.add_scalar('Train/Loss', avgTrainLoss, e + 1)
            writer.add_scalar('Train/accu', avgTrainAccu, e + 1)
            writer.add_scalar('Valid/loss', avgValidationLoss, e + 1)
            writer.add_scalar('Valid/accu', avgValidationAccu, e + 1)
            writer.flush()
            # 4.8、save best_validation_dsc model params
            if avgValidationAccu > best_validation_dsc:
                best_validation_dsc = avgValidationAccu
                # best_model_params = self.model.state_dict()
                # serialize best model to disk
                torch.save(self.model.state_dict(), MODEL_PATH)
        # display the total time needed to perform the training
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
        # 5、plot the training loss
        plot_result(model_dir, H["train_loss"], H["valdation_loss"], "train_loss", "valdation_loss", "loss")
        plot_result(model_dir, H["train_accuracy"], H["valdation_accuracy"], "train_accuracy", "valdation_accuracy",
                    "accuracy")
        self.clear_GPU_cache()

    def predict(self, full_img, out_threshold=0.5):
        # 1、clear cache
        self.clear_GPU_cache()
        # 2、set model eval
        self.model.eval()
        # 3、convet numpy image to tensor
        img = torch.as_tensor(full_img).float().contiguous()
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)
        # 4、predict result
        with torch.no_grad():
            output = self.model(img)
            if self.numclass == 1:
                probs = torch.sigmoid(output[0])
            else:
                probs = torch.softmax(output[0], dim=0)
            full_mask_np = probs.detach().cpu().squeeze().numpy()
        # 5、get numpy result
        if self.numclass == 1:
            out_mask = (full_mask_np > out_threshold)
            out_mask = out_mask * 255
        else:
            out_mask = np.argmax(full_mask_np, axis=0)
            out_mask = np.squeeze(out_mask)
        return out_mask

    def inference(self, image):
        # resize image and normalization
        imageresize = cv2.resize(image, (self.image_width, self.image_height))
        imageresize = imageresize / 255.
        # transpose (H,W,C) order to (C,H,W) order
        H, W = np.shape(imageresize)[0], np.shape(imageresize)[1]
        imageresize = np.reshape(imageresize, (H, W, 1))
        imageresize = np.transpose(imageresize, (2, 0, 1))
        out_mask = self.predict(imageresize)
        return out_mask

    def clear_GPU_cache(self):
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())


class MutilResNet2dModel(object):
    """
    ResNet2d with mutil class,should rewrite the dataset class and inference fucntion
    """

    def __init__(self, image_height, image_width, image_channel, numclass, batch_size, loss_name='MutilFocalLoss',
                 inference=False, model_path=None, use_cuda=True):
        self.batch_size = batch_size
        self.loss_name = loss_name
        self.accuracyname = 'accu'
        self.image_height = image_height
        self.image_width = image_width
        self.image_channel = image_channel
        self.numclass = numclass

        self.alpha = [1.] * self.numclass
        self.gamma = 2

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = ResNet2d(self.image_channel, self.numclass)
        self.model.to(device=self.device)
        self.alpha = torch.as_tensor(self.alpha).contiguous().to(self.device)
        if inference:
            print(f'Loading model {model_path}')
            print(f'Using device {self.device}')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded!')

    def _dataloder(self, images, labels, shuffle=False):
        """"
        train dataset shuffle is true,validation is false
        """
        # Number of workers
        dataset = datasetModelClassifywithopencv(images, labels,
                                                 targetsize=(self.image_channel, self.image_height, self.image_width))
        # fow window num_workers is only zero,for linux num_workers can not zero
        # num_cpu = multiprocessing.cpu_count()
        num_cpu = 0
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=self.batch_size, num_workers=num_cpu,
                                pin_memory=True)
        return dataloader

    def _loss_function(self, lossname):
        if lossname is 'MutilCrossEntropyLoss':
            return MutilCrossEntropyLoss(self.alpha)
        if lossname is 'MutilFocalLoss':
            return MutilFocalLoss(self.alpha, self.gamma)

    def _accuracy_function(self, accuracyname, input, target):
        if accuracyname is 'accu':
            if self.numclass == 1:
                input = (input > 0.5).float()
                target = (target > 0.5).float()
                return calc_accuracy(input, target)
            else:
                input = torch.argmax(input, 1)
                return calc_accuracy(input, target)

    def trainprocess(self, trainimage, trainmask, validationimage, validationmask, model_dir, epochs=50, lr=1e-3):
        print("[INFO] training the network...")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        MODEL_PATH = os.path.join(model_dir, "MutilResNet2d.pth")
        summary(self.model, input_size=(self.image_channel, self.image_height, self.image_width))
        print(self.model)
        # 1、initialize loss function and optimizer
        lossFunc = self._loss_function(self.loss_name)
        opt = optim.Adam(self.model.parameters(), lr=lr)
        # 2、load data train and validation dataset
        train_loader = self._dataloder(trainimage, trainmask, True)
        val_loader = self._dataloder(validationimage, validationmask)
        # 3、initialize a dictionary to store training history
        H = {"train_loss": [], "train_accuracy": [], "valdation_loss": [], "valdation_accuracy": []}
        # 4、start loop training wiht epochs times
        startTime = time.time()
        best_validation_dsc = 0.0
        # Tensorboard summary
        writer = SummaryWriter(log_dir=model_dir)
        for e in tqdm(range(epochs)):
            # 4.1、set the model in training mode
            self.model.train()
            # 4.2、initialize the total training and validation loss
            totalTrainLoss = []
            totalTrainAccu = []
            totalValidationLoss = []
            totalValiadtionAccu = []
            # 4.3、loop over the training set
            for batch in train_loader:
                # x should tensor with shape (N,C,W,H)
                x = batch['image']
                # y should tensor with shape (N,C,W,H),
                # if have mutil label y should one-hot,if only one label,the C is one
                y = batch['label']
                # send the input to the device
                x, y = x.to(self.device), y.to(self.device)
                # perform a forward pass and calculate the training loss and accu
                pred_logits = self.model(x)
                loss = lossFunc(pred_logits, y)
                # first, zero out any previously accumulated gradients,
                # then perform backpropagation,
                # and then update model parameters
                opt.zero_grad()
                loss.backward()
                opt.step()
                pred = F.softmax(pred_logits, dim=1)
                accu = self._accuracy_function(self.accuracyname, pred, y)
                # add the loss to the total training loss so far
                totalTrainLoss.append(loss)
                totalTrainAccu.append(accu)
            # 4.4、switch off autograd and loop over the validation set
            with torch.no_grad():
                # set the model in evaluation mode
                self.model.eval()
                # loop over the validation set
                for batch in val_loader:
                    # x should tensor with shape (N,C,W,H)
                    x = batch['image']
                    # y should tensor with shape (N,)
                    y = batch['label']
                    # send the input to the device
                    (x, y) = (x.to(self.device), y.to(self.device))
                    # make the predictions and calculate the validation loss
                    pred_logits = self.model(x)
                    loss = lossFunc(pred_logits, y)
                    # save_images
                    pred = F.softmax(pred_logits, dim=1)
                    accu = self._accuracy_function(self.accuracyname, pred, y)
                    totalValidationLoss.append(loss)
                    totalValiadtionAccu.append(accu)
            # 4.5、calculate the average training and validation loss
            avgTrainLoss = torch.mean(torch.stack(totalTrainLoss))
            avgValidationLoss = torch.mean(torch.stack(totalValidationLoss))
            avgTrainAccu = torch.mean(torch.stack(totalTrainAccu))
            avgValidationAccu = torch.mean(torch.stack(totalValiadtionAccu))
            # 4.6、update our training history
            H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            H["valdation_loss"].append(avgValidationLoss.cpu().detach().numpy())
            H["train_accuracy"].append(avgTrainAccu.cpu().detach().numpy())
            H["valdation_accuracy"].append(avgValidationAccu.cpu().detach().numpy())
            # 4.7、print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
            print("Train loss: {:.5f}, Train accu: {:.5f}，validation loss: {:.5f}, validation accu: {:.5f}".format(
                avgTrainLoss, avgTrainAccu, avgValidationLoss, avgValidationAccu))
            # Record training loss and accuracy for each phase
            writer.add_scalar('Train/Loss', avgTrainLoss, e + 1)
            writer.add_scalar('Train/accu', avgTrainAccu, e + 1)
            writer.add_scalar('Valid/loss', avgValidationLoss, e + 1)
            writer.add_scalar('Valid/accu', avgValidationAccu, e + 1)
            writer.flush()
            # 4.8、save best_validation_dsc model params
            if avgValidationAccu > best_validation_dsc:
                best_validation_dsc = avgValidationAccu
                # best_model_params = self.model.state_dict()
                # serialize best model to disk
                torch.save(self.model.state_dict(), MODEL_PATH)
        # display the total time needed to perform the training
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
        # 5、plot the training loss
        plot_result(model_dir, H["train_loss"], H["valdation_loss"], "train_loss", "valdation_loss", "loss")
        plot_result(model_dir, H["train_accuracy"], H["valdation_accuracy"], "train_accuracy", "valdation_accuracy",
                    "accuracy")
        self.clear_GPU_cache()

    def predict(self, full_img, out_threshold=0.5):
        # 1、clear cache
        self.clear_GPU_cache()
        # 2、set model eval
        self.model.eval()
        # 3、convet numpy image to tensor
        img = torch.as_tensor(full_img).float().contiguous()
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)
        # 4、predict result
        with torch.no_grad():
            output = self.model(img)
            if self.numclass == 1:
                probs = torch.sigmoid(output[0])
            else:
                probs = torch.softmax(output[0], dim=0)
            full_mask_np = probs.detach().cpu().squeeze().numpy()
        # 5、get numpy result
        if self.numclass == 1:
            out_mask = (full_mask_np > out_threshold)
            out_mask = out_mask * 255
        else:
            out_mask = np.argmax(full_mask_np, axis=0)
            out_mask = np.squeeze(out_mask)
        return out_mask

    def inference(self, image):
        # resize image and normalization
        imageresize = cv2.resize(image, (self.image_width, self.image_height))
        imageresize = imageresize / 255.
        # transpose (H,W,C) order to (C,H,W) order
        H, W = np.shape(imageresize)[0], np.shape(imageresize)[1]
        imageresize = np.reshape(imageresize, (H, W, 1))
        imageresize = np.transpose(imageresize, (2, 0, 1))
        out_mask = self.predict(imageresize)
        return out_mask

    def clear_GPU_cache(self):
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())


class BinaryResNet3dModel(object):
    """
    ResNet3d with binary class,should rewrite the dataset class and inference fucntion
    """

    def __init__(self, image_depth, image_height, image_width, image_channel, numclass, batch_size,
                 loss_name='BinaryCrossEntropyLoss', inference=False, model_path=None, use_cuda=True):
        self.batch_size = batch_size
        self.loss_name = loss_name
        self.accuracyname = 'accu'
        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.image_channel = image_channel
        self.numclass = numclass

        self.alpha = 0.25
        self.gamma = 2

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = ResNet3d(self.image_channel, self.numclass)
        self.model.to(device=self.device)

        if inference:
            print(f'Loading model {model_path}')
            print(f'Using device {self.device}')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded!')

    def _dataloder(self, images, labels, shuffle=False):
        """"
        train dataset shuffle is true,validation is false
        """
        dataset = datasetModelClassifywithnpy(images, labels,
                                              targetsize=(
                                                  self.image_channel, self.image_depth, self.image_height,
                                                  self.image_width))
        # fow window num_workers is only zero,for linux num_workers can not zero
        # num_cpu = multiprocessing.cpu_count()
        num_cpu = 0
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=self.batch_size, num_workers=num_cpu,
                                pin_memory=True)
        return dataloader

    def _loss_function(self, lossname):
        if lossname is 'BinaryCrossEntropyLoss':
            return BinaryCrossEntropyLoss()
        if lossname is 'BinaryFocalLoss':
            return BinaryFocalLoss(alpha=self.alpha, gamma=self.gamma)

    def _accuracy_function(self, accuracyname, input, target):
        if accuracyname is 'accu':
            if self.numclass == 1:
                input = (input > 0.5).float()
                target = (target > 0.5).float()
                return calc_accuracy(input, target)
            else:
                input = torch.argmax(input, 1)
                return calc_accuracy(input, target)

    def trainprocess(self, trainimage, trainmask, validationimage, validationmask, model_dir, epochs=50, lr=1e-3):
        print("[INFO] training the network...")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        MODEL_PATH = os.path.join(model_dir, "BinaryResNet3d.pth")
        summary(self.model, input_size=(self.image_channel, self.image_depth, self.image_height, self.image_width))
        print(self.model)
        # 1、initialize loss function and optimizer
        lossFunc = self._loss_function(self.loss_name)
        opt = optim.Adam(self.model.parameters(), lr=lr)
        #lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=2, verbose=True)
        # 2、load data train and validation dataset
        train_loader = self._dataloder(trainimage, trainmask, True)
        val_loader = self._dataloder(validationimage, validationmask)
        # 3、initialize a dictionary to store training history
        H = {"train_loss": [], "train_accuracy": [], "valdation_loss": [], "valdation_accuracy": []}
        # 4、start loop training wiht epochs times
        startTime = time.time()
        best_validation_dsc = 0.0
        # Tensorboard summary
        writer = SummaryWriter(log_dir=model_dir)
        for e in tqdm(range(epochs)):
            # 4.1、set the model in training mode
            self.model.train()
            # 4.2、initialize the total training and validation loss
            totalTrainLoss = []
            totalTrainAccu = []
            totalValidationLoss = []
            totalValiadtionAccu = []
            # 4.3、loop over the training set
            for batch in train_loader:
                # x should tensor with shape (N,C,D,W,H)
                x = batch['image']
                # y should tensor with shape (N,C,D,W,H),
                # if have mutil label y should one-hot,if only one label,the C is one
                y = batch['label']
                # send the input to the device
                x, y = x.to(self.device), y.to(self.device)
                # perform a forward pass and calculate the training loss and accu
                pred_logit = self.model(x)
                loss = lossFunc(pred_logit, y)
                pred = F.sigmoid(pred_logit)
                accu = self._accuracy_function(self.accuracyname, pred, y)
                # first, zero out any previously accumulated gradients,
                # then perform backpropagation,
                # and then update model parameters
                opt.zero_grad()
                loss.backward()
                opt.step()
                # add the loss to the total training loss so far
                totalTrainLoss.append(loss)
                totalTrainAccu.append(accu)
            # 4.4、switch off autograd and loop over the validation set
            with torch.no_grad():
                # set the model in evaluation mode
                self.model.eval()
                # loop over the validation set
                for batch in val_loader:
                    # x should tensor with shape (N,C,D,W,H)
                    x = batch['image']
                    # y should tensor with shape (N,)
                    y = batch['label']
                    # send the input to the device
                    (x, y) = (x.to(self.device), y.to(self.device))
                    # make the predictions and calculate the validation loss
                    pred_logit = self.model(x)
                    loss = lossFunc(pred_logit, y)
                    pred = F.sigmoid(pred_logit)
                    accu = self._accuracy_function(self.accuracyname, pred, y)
                    totalValidationLoss.append(loss)
                    totalValiadtionAccu.append(accu)
            # 4.5、calculate the average training and validation loss
            avgTrainLoss = torch.mean(torch.stack(totalTrainLoss))
            avgValidationLoss = torch.mean(torch.stack(totalValidationLoss))
            avgTrainAccu = torch.mean(torch.stack(totalTrainAccu))
            avgValidationAccu = torch.mean(torch.stack(totalValiadtionAccu))
            #lr_scheduler.step(avgValidationLoss)
            # 4.6、update our training history
            H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            H["valdation_loss"].append(avgValidationLoss.cpu().detach().numpy())
            H["train_accuracy"].append(avgTrainAccu.cpu().detach().numpy())
            H["valdation_accuracy"].append(avgValidationAccu.cpu().detach().numpy())
            # 4.7、print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
            print("Train loss: {:.5f}, Train accu: {:.5f}，validation loss: {:.5f}, validation loss: {:.5f}".format(
                avgTrainLoss, avgTrainAccu, avgValidationLoss, avgValidationAccu))
            # Record training loss and accuracy for each phase
            writer.add_scalar('Train/Loss', avgTrainLoss, e + 1)
            writer.add_scalar('Train/accu', avgTrainAccu, e + 1)
            writer.add_scalar('Valid/loss', avgValidationLoss, e + 1)
            writer.add_scalar('Valid/accu', avgValidationAccu, e + 1)
            writer.flush()
            # 4.8、save best_validation_dsc model params
            if avgValidationAccu > best_validation_dsc:
                best_validation_dsc = avgValidationAccu
                # best_model_params = self.model.state_dict()
                # serialize best model to disk
                torch.save(self.model.state_dict(), MODEL_PATH)
        # display the total time needed to perform the training
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
        # 5、plot the training loss
        plot_result(model_dir, H["train_loss"], H["valdation_loss"], "train_loss", "valdation_loss", "loss")
        plot_result(model_dir, H["train_accuracy"], H["valdation_accuracy"], "train_accuracy", "valdation_accuracy",
                    "accuracy")
        self.clear_GPU_cache()

    def predict(self, full_img, out_threshold=0.5):
        # 1、clear cache
        self.clear_GPU_cache()
        # 2、set model eval
        self.model.eval()
        # 3、convet numpy image to tensor
        img = torch.as_tensor(full_img).float().contiguous()
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)
        # 4、predict result
        with torch.no_grad():
            output = self.model(img)
            if self.numclass == 1:
                probs = torch.sigmoid(output[0])
            else:
                probs = torch.softmax(output[0], dim=0)
            full_mask_np = probs.detach().cpu().squeeze().numpy()
        # 5、get numpy result
        if self.numclass == 1:
            out_mask = (full_mask_np > out_threshold)
            out_mask = out_mask * 255
        else:
            out_mask = np.argmax(full_mask_np, axis=0)
            out_mask = np.squeeze(out_mask)
        return out_mask

    def inference(self, image):
        # resize image and normalization,should rewrite
        imageresize = image
        # transpose (D,H,W,C) order to (C,D,H,W) order
        D, H, W = np.shape(imageresize)[0], np.shape(imageresize)[1], np.shape(imageresize)[2]
        imageresize = np.reshape(imageresize, (D, H, W, 1))
        imageresize = np.transpose(imageresize, (3, 0, 1, 2))
        out_mask = self.predict(imageresize)
        # resize mask to src image size,should rewrite
        return out_mask

    def clear_GPU_cache(self):
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())


class MutilResNet3dModel_metric(object):
    """
    ResNet3d with mutil class,should rewrite the dataset class
    """

    def __init__(self, image_depth, image_height, image_width, image_channel, numclass, batch_size,
                 loss_name='MutilFocalLoss', inference=False, model_path=None, use_cuda=True):
        self.batch_size = batch_size
        self.loss_name = loss_name
        self.accuracyname = 'accu'
        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.image_channel = image_channel
        self.numclass = numclass

        self.alpha = [1.] * self.numclass
        self.gamma = 2

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = VGG3d(self.image_channel, self.numclass)
        self.model.to(device=self.device)
        self.alpha = torch.as_tensor(self.alpha).contiguous().to(self.device)
        if inference:
            print(f'Loading model {model_path}')
            print(f'Using device {self.device}')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded!')

    def _dataloder(self, images, labels, shuffle=False):
        """"
        train dataset shuffle is true,validation is false
        """
        dataset = datasetModelClassifywithnpy(images, labels,
                                              targetsize=(
                                                  self.image_channel, self.image_depth, self.image_height,
                                                  self.image_width))
        # fow window num_workers is only zero,for linux num_workers can not zero
        # num_cpu = multiprocessing.cpu_count()
        num_cpu = 0
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=self.batch_size, num_workers=num_cpu,
                                pin_memory=True)
        return dataloader

    def _loss_function(self, lossname):
        if lossname == 'MutilCrossEntropyLoss':
            return MutilCrossEntropyLoss(self.alpha)
        if lossname == 'MutilFocalLoss':
            return MutilFocalLoss(self.alpha, self.gamma)

    def _accuracy_function(self, accuracyname, input, target):
        if accuracyname is 'accu':
            if self.numclass == 1:
                input = (input > 0.5).float()
                target = (target > 0.5).float()
                return calc_accuracy(input, target)
            else:
                input = torch.argmax(input, 1)  # 返回指定维度最大的序号
                return calc_accuracy(input, target)

    def trainprocess(self, trainimage, trainmask, validationimage, validationmask, model_dir, epochs=50, lr=1e-3):
        print("[INFO] training the network...")
        # Path(model_dir).mkdir(parents=True, exist_ok=True)
        # MODEL_PATH = os.path.join(model_dir, "MutilResNet3d.pth")
        summary(self.model, input_size=(self.image_channel, self.image_depth, self.image_height, self.image_width))
        print(self.model)
        # 1、initialize loss function and optimizer
        lossFunc = self._loss_function(self.loss_name)
        opt = optim.AdamW(self.model.parameters(), lr=lr)
        # 2、load data train and validation dataset
        train_loader = self._dataloder(trainimage, trainmask, True)
        val_loader = self._dataloder(validationimage, validationmask)
        # 3、initialize a dictionary to store training history
        H = {"train_loss": [], "train_accuracy": [], "valdation_loss": [], "valdation_accuracy": []}
        # 4、start loop training wiht epochs times
        startTime = time.time()
        best_validation_dsc = 0.0
        epoch_of_best_validation_acc = 0
        # Tensorboard summary
        # writer = SummaryWriter(log_dir=model_dir)
        # 4.1、set the model in training mode
        self.model.train()
        # 4.2、initialize the total training and validation loss
        totalTrainLoss = []
        totalTrainAccu = []
        totalValidationLoss = []
        totalValiadtionAccu = []
        total_pred_y = []
        total_y = []
        total_pred_score = []

        conf_maxtri = np.ndarray((3, 3))

        # 4.4、switch off autograd and loop over the validation set
        with torch.no_grad():
            # set the model in evaluation mode
            # self.model.eval()
            # loop over the validation set
            for batch in val_loader:
                # x should tensor with shape (N,C,W,H)
                x = batch['image']
                # y should tensor with shape (N,C,W,H)
                y = batch['label']
                # send the input to the device
                (x, y) = (x.to(self.device), y.to(self.device))
                # make the predictions and calculate the validation loss
                pred_logit = self.model(x)
                loss = lossFunc(pred_logit, y)
                pred = F.softmax(pred_logit, dim=1)
                pred_y = torch.argmax(pred, dim=1)
                # pred_score, pred_y = torch.max(pred, dim=1)
                ###

                ###
                accu = self._accuracy_function(self.accuracyname, pred, y)
                totalValidationLoss.append(loss)
                totalValiadtionAccu.append(accu)
                total_y.extend(y.cpu().tolist())
                total_pred_y.extend(pred_y.cpu().tolist())
                total_pred_score.extend(pred.cpu().tolist())

            total_y = torch.tensor(total_y, dtype=torch.long)
            total_y = total_y.unsqueeze(-1)
            total_y = torch.zeros(total_y.shape[0], 3).scatter_(1, total_y, 1)
            total_y = total_y.cpu().numpy()
            total_pred_score = np.array(total_pred_score)

            total_pred_y = torch.tensor(total_pred_y, dtype=torch.long)
            total_pred_y = total_pred_y.unsqueeze(-1)
            total_pred_y = torch.zeros(total_pred_y.shape[0], 3).scatter_(1, total_pred_y, 1)
            total_pred_y = total_pred_y.cpu().numpy()

            # 4.5、calculate the average training and validation loss
            # avgTrainLoss = torch.mean(torch.stack(totalTrainLoss))
            avgValidationLoss = torch.mean(torch.stack(totalValidationLoss))
            # avgTrainAccu = torch.mean(torch.stack(totalTrainAccu))
            avgValidationAccu = torch.mean(torch.stack(totalValiadtionAccu))
            # 4.6、update our training history
            # H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            H["valdation_loss"].append(avgValidationLoss.cpu().detach().numpy())
            # H["train_accuracy"].append(avgTrainAccu.cpu().detach().numpy())
            H["valdation_accuracy"].append(avgValidationAccu.cpu().detach().numpy())
            # 4.7、print the model training and validation information

            print("validation loss: {:.5f}, validation accu: {:.5f}".format(
                avgValidationLoss, avgValidationAccu))

            from sklearn.metrics import roc_curve, precision_recall_curve, auc
            from itertools import cycle
            # 绘制ROC曲线
            # 计算每一类的ROC
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            all_roc_auc = 0

            for i in range(self.numclass):
                fpr[i], tpr[i], _ = roc_curve(total_y[:, i], total_pred_y[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                # roc_auc累加值
                all_roc_auc += roc_auc[i]
            print('average_roc_auc: ', all_roc_auc/self.numclass)

            # Plot all ROC curves
            lw = 2
            plt.figure()
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
            for i, color in zip(range(self.numclass), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                         label='ROC curve of class {0} (area = {1:0.4f})'
                               ''.format(i, roc_auc[i]))

            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Malignancy-calss ROC')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(model_dir, 'Malignancy-calss ROC{}.png'.format(all_roc_auc/self.numclass)))
            plt.show()

            ## 绘制pr曲线
            # 计算每一类的pr
            precision = dict()
            recall = dict()
            pr_auc = dict()
            all_pr_auc = 0
            for i in range(self.numclass):
                precision[i], recall[i], _ = precision_recall_curve(total_y[:, i], total_pred_y[:, i])
                pr_auc[i] = auc(recall[i], precision[i])
                # pr_auc累加值
                all_pr_auc += pr_auc[i]
            print('average_pr_auc: ', all_pr_auc / self.numclass)

            # Plot all PR curves
            lw = 2
            plt.figure()
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
            for i, color in zip(range(self.numclass), colors):
                plt.plot(recall[i], precision[i], color=color, lw=lw,
                         label='PR curve of class {0} (area = {1:0.4f})'
                               ''.format(i, pr_auc[i]))

            # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Malignancy-calss PR')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(model_dir, 'Malignancy-calss PR{}.png'.format(all_pr_auc/self.numclass)))
            plt.show()

            ## 混淆矩阵可视化
            # import seaborn as sns
            # from sklearn import metrics
            # import pandas as pd
            #
            # cf_matrix = metrics.confusion_matrix(total_y, total_pred_y)
            # sum_true = np.expand_dims(np.sum(cf_matrix, axis=1), axis=1)
            # precision_matrix = cf_matrix / sum_true
            # df = pd.DataFrame(precision_matrix)
            # ax = sns.heatmap(df, cmap="Blues", annot=True)
            # ax.set_title('confusion matrix')
            # ax.set_xlabel('predict')
            # ax.set_ylabel('true')
            # plt.savefig(os.path.join(model_dir, 'Texture_cf_matrix.png'))
            # plt.show()
            #
            # # 分类评估指标
            # from sklearn.metrics import classification_report
            # report = classification_report(total_y, total_pred_y)
            # print(report)

            '''## 绘制ROC曲线
            # 用metrics.roc_curve()求出 fpr, tpr, threshold
            fpr0, tpr0, threshold0 = metrics.roc_curve(total_y, total_pred_score, pos_label=0)
            fpr1, tpr1, threshold1 = metrics.roc_curve(total_y, total_pred_score, pos_label=1)
            fpr2, tpr2, threshold2 = metrics.roc_curve(total_y, total_pred_score, pos_label=2)

            # 用metrics.auc求出roc_auc的值
            roc_auc0 = metrics.auc(fpr0, tpr0)
            roc_auc1 = metrics.auc(fpr1, tpr1)
            roc_auc2 = metrics.auc(fpr2, tpr2)

            # 将plt.plot里的内容填写完整
            plt.plot(fpr0, tpr0, label='AUC = %0.2f' % roc_auc0)
            plt.plot(fpr1, tpr1, label='AUC = %0.2f' % roc_auc1)
            plt.plot(fpr2, tpr2, label='AUC = %0.2f' % roc_auc2)

            # 将图例显示在右下方
            plt.legend(loc='lower right')

            # 画出一条红色对角虚线
            plt.plot([0, 1], [0, 1], 'r--')

            # 设置横纵坐标轴范围
            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])

            # 设置横纵名称以及图形名称
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.title('Receiver Operating Characteristic Curve')
            plt.show()'''

            '''## 绘制pr曲线
            # 用metrics.roc_curve()求出 fpr, tpr, threshold
            precision0, recall0, threshold0 = metrics.precision_recall_curve(total_y, total_pred_score, pos_label=0)
            precision1, recall1, threshold1 = metrics.precision_recall_curve(total_y, total_pred_score, pos_label=1)
            precision2, recall2, threshold2 = metrics.precision_recall_curve(total_y, total_pred_score, pos_label=2)

            # 用metrics.auc求出roc_auc的值
            pr_auc0 = metrics.auc(recall0, precision0)
            pr_auc1 = metrics.auc(recall1, precision1)
            pr_auc2 = metrics.auc(recall2, precision2)

            # 将plt.plot里的内容填写完整
            plt.plot(recall0, precision0, label='AUC = %0.2f' % pr_auc0)
            plt.plot(recall1, precision1, label='AUC = %0.2f' % pr_auc1)
            plt.plot(recall2, precision2, label='AUC = %0.2f' % pr_auc2)

            # 将图例显示在右下方
            plt.legend(loc='lower right')

            # 画出一条红色对角虚线
            plt.plot([0, 1], [0, 1], 'r--')

            # 设置横纵坐标轴范围
            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])

            # 设置横纵名称以及图形名称
            plt.ylabel('Precision')
            plt.xlabel('Recall')
            plt.title('Receiver Operating Characteristic Curve')
            plt.show()'''

            # print("Train loss: {:.5f}, Train accu: {:.5f}，validation loss: {:.5f}, validation accu: {:.5f}".format(
            #     avgTrainLoss, avgTrainAccu, avgValidationLoss, avgValidationAccu))

        # display the total time needed to perform the training
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))

        self.clear_GPU_cache()


    def predict(self, full_img, out_threshold=0.5):
        # 1、clear cache
        self.clear_GPU_cache()
        # 2、set model eval
        self.model.eval()
        # 3、convet numpy image to tensor
        img = torch.as_tensor(full_img).float().contiguous()
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)
        # 4、predict result
        with torch.no_grad():
            output = self.model(img)
            if self.numclass == 1:
                probs = torch.sigmoid(output[0])
            else:
                probs = torch.softmax(output[0], dim=0)
            full_mask_np = probs.detach().cpu().squeeze().numpy()
        # 5、get numpy result
        if self.numclass == 1:
            out_mask = (full_mask_np > out_threshold)
            out_mask = out_mask * 255
        else:
            out_mask = np.argmax(full_mask_np, axis=0)
            out_mask = np.squeeze(out_mask)
        return out_mask

    def inference(self, image):
        # resize image and normalization,should rewrite
        imageresize = image
        # transpose (D,H,W,C) order to (C,D,H,W) order
        D, H, W = np.shape(imageresize)[0], np.shape(imageresize)[1], np.shape(imageresize)[2]
        imageresize = np.reshape(imageresize, (D, H, W, 1))
        imageresize = np.transpose(imageresize, (3, 0, 1, 2))
        out_mask = self.predict(imageresize)
        # resize mask to src image size,shou rewrite
        return out_mask

    def clear_GPU_cache(self):
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())
