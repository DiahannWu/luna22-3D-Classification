

file_name = 'data_Malignancy/traindata.csv'

labels = [0, 1]

for label in labels:
    print('===============gen traindata{}.csv=================='.format(label))
    csvfile = open(file_name, 'r')
    with open('data_Malignancy/traindata{}.csv'.format(label), 'w+') as csv_file2:
        csv_file2.writelines("Image,Mask" + "\n")
        count = 0
        for row in csvfile:
            if str(',{}'.format(label)) in row:
                csv_file2.write(row)
                count += 1
        print('count:', count)

## 统计各类别个数
'''file_name = 'data_new/alldata.csv'

labels = [0, 1, 2]

for label in labels:
    print('===============count of alldata.csv in label{}=================='.format(label))
    csvfile = open(file_name, 'r')
    count = 0
    for row in csvfile:
        if str(',{}'.format(label)) in row:
            count += 1
    print('count:', count)'''