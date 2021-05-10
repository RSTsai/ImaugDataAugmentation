

def CheckDir(dirPath):
    from os import makedirs, path

    try:
        if path.exists(dirPath):
            return True
        else:
            print("路徑不存在。")
            print(f"建立路徑:{dirPath}")
            makedirs(dirPath)
            return True
    except FileExistsError:
        print("路徑已存在")
        return True
    except PermissionError:
        # 權限不足的例外處理
        print("權限不足")
        return False


def CheckFile(filePath):
    from os import path

    try:
        if path.exists(filePath):
            return True
        else:
            print(f"路徑不存在{filePath}")
    except FileExistsError:
        print("路徑已存在")
        return True
    except PermissionError:
        # 權限不足的例外處理
        print("權限不足")
        return False
