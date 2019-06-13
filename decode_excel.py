# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 16:20:32 2018

@author: Administrator
"""

import xlrd, os, re
plate = ['粤A6503H', '粤A177US', '粤A3659D', '粤A0RE33', '粤A4102J', '粤A5RW15', '粤A9CE55', 
 '粤AC777X', '粤A7AB78', '粤A8716Z', '粤AY658X', '粤A8KS92', '粤A0TH69', '粤A5HR14', 
 '粤AL372R', '粤A0GM68', '粤A973FS', '粤A6860U', '粤A205ZE', '粤A2KC31']
def excel2dic(file_path):
    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_name('Sheet1')
    nrows = table.nrows
    print(nrows)
    result = []
    for i in range(1, 21):
        row_value = table.row_values(i)
        label = row_value[0]
        #image_name = row_value[2]
        result.append(label)
        
    return result

def construct_path(path):
    try:
        files = os.listdir(path)
        for subfile in files:
            subfile_path = os.path.join(path, subfile)
            for element in construct_path(subfile_path):
                _, ext = os.path.splitext(element)
                if ext in ['.jpg', '.JPG']:
                    yield element
    except (TypeError, NotADirectoryError):
        _, ext = os.path.splitext(path)
        if ext in ['.jpg', '.JPG']:
            yield path
            
def cons_gt(path):
    r = construct_path(path)
    paths = list(r)
    output = []
    for path in paths:
        basename = os.path.basename(path)
        name, _ = os.path.splitext(basename)
        if not re.search(r"_", name):
            output.append(name)
        else:
            #print(name)
            num = int(re.findall(r'.*_(.*)', name)[0])
            output.append(plate[num - 1])
        
    return paths, output
if __name__ == '__main__':
    paths, temp = cons_gt(r'F:\licensePlateRecognition\data\test-set\img-test')
    print(temp[1000], paths[1000])
    print(len(temp))