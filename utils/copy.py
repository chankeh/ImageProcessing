import shutil
import os

# get file list in source dir
def get_file_list(src_dir):
    file_list = []
    for filename in os.listdir(src_dir):
        file_list.append(filename)
        # print(filename)
    return file_list

# copy source files paths in plane txt file to target dir
def copy_files(src_dir, dst_dir):
    file_index = 0
    file_list = get_file_list(src_dir)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    for file in file_list:
        tmp_file_path = src_dir + file
        dst_file_path = dst_dir + file
        if os.path.isfile(tmp_file_path):
            shutil.copyfile(tmp_file_path, dst_file_path)
            # print(dst_file_path)
            file_index += 1
    # print(file_index)

# copy souce dir to dst_dir
def copy_dir_files(src_dir, dst_dir, filter=True):
    file_list = get_file_list(src_dir)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    for file in file_list:
        tmp_file_path = src_dir + file
        dst_file_path = dst_dir + file
        if os.path.isfile(tmp_file_path):
            shutil.copyfile(tmp_file_path, dst_file_path)
            # print(dst_file_path)
        if os.path.isdir(tmp_file_path):
            # filter to skip file or folder like ".*"
            # print(tmp_file_path)
            if filter == True:
                if file[0] == '.':
                    continue
                copy_dir_files(tmp_file_path + '/', dst_file_path + '/', filter=True)


if __name__ == "__main__":
    src_dir = "./"
    dst_dir = "./tmp/"
    copy_files(src_dir, dst_dir)
    copy_dir_files(src_dir, dst_dir, filter=True)