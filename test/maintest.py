import cv2
import os

from test import timbienso
from tachchar import tachkytu
from testread import docbien


def main():
    path_img='datatestbienso/427.jpg'
    img=cv2.imread(path_img)
    bienso=timbienso(img)
    tachkytu(bienso)
    cv2.imshow('bien so xe',bienso)
    # print('bien so la ',docbien())
    folder='kytucut'
    # text = docbien()
    # text = "51A12345"
    # Ghi chữ lên góc trên bên trái
    cv2.putText(img,docbien(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,  (0, 0, 255),  2, cv2.LINE_AA)
    cv2.imshow('img',img)
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Lỗi khi xóa {file_path}: {e}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()