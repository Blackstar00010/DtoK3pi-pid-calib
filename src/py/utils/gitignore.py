import os
from src.utils import utils


def main():
    # list all files in the current directory that are larger than 80MB
    with open('.gitignore', 'r') as f:
        gitignore = f.read()
    ls = utils.listdir('.', show_hidden=True, recursive=True)
    for file in ls:
        filesize = os.path.getsize(file) / 1024 / 1024  # in MB
        if filesize > 80 and file not in gitignore:
            print(f"{file} has file size of {filesize:.2f} MB")
            res = input('Would you like to add this file to gitignore? (y/n): ')
            if res.lower() == 'y':
                gitignore += f"\n{file}"

    with open('.gitignore', 'w') as f:
        f.write(gitignore)


if __name__ == "__main__":
    main()
