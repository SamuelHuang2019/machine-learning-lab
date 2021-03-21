path = '../../data/VOC2007/'
with open(path + 'label_train.txt', 'w') as fw:
    for i in range(1, 201):
        file_name = path + 'labels/' + str(i).rjust(7, '0') + '.txt'
        with open(file_name, 'r') as fr:
            for line in fr.readlines():
                fw.write(line.strip() + ' ')
            fw.write('\n')
