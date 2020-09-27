import os
#train텍스트 파일은 6115개
#trainval 텍스트 파일은 7954
#val 텍스트 파일은 1839
val_text = open("val.txt", "r")
val_content = val_text.read()
train_text = open('train.txt', 'r')
train_content = train_text.read()

val_list = val_content.split('\n')
train_list = train_content.split('\n')

intersection_set = set.intersection(set(val_list), set(train_list))

intersection_list = list(intersection_set)
print(intersection_list)