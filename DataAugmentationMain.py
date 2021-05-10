import os
import sys
import glob
import cv2
import pathlib
import xml.etree.ElementTree as ET
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import matplotlib.pyplot as plt
from OsFunction.OsFileModel import CheckDir
from DataAugmentation.ImaugFunctionsModel import ImaugFunctions




# imput picture parameter
imgFolderPath = f'./VideoToPicture/JPEGImages/' 
xmlFolderPath = f'./VideoToPicture/Annotations/'
fileExtensions = [ "jpg", "jpeg", "png", "bmp", "gif" ]
# output picture parameter
saveRootPath = f'./DataAugmentation/'
saveImgFolderName = "JPEGImages"
saveXmlFolderName = "Annotations"
batchNum = 1
batchRunNum = 10

def DataAugmentationFlow(seq=None):
    # FilesList
    imgFilesList = []
    for extension in fileExtensions:
        imgFilesList.extend( glob.glob( os.path.join(imgFolderPath ,f'*.{extension}') ))

    # save Folder Create
    saveImgFolderPath = os.path.join(saveRootPath, f"NN_960_DataAugmentation#{batchNum}", saveImgFolderName)
    if CheckDir(saveImgFolderPath) == False:
        return
    saveXmlFolderPath = os.path.join(saveRootPath, f"NN_960_DataAugmentation#{batchNum}", saveXmlFolderName)
    if CheckDir(saveXmlFolderPath) == False:
        return

 
    


    # img processsing
    for item in imgFilesList:
        # path set
        imgPath = pathlib.Path(item)  
        saveImgFilePath = os.path.join(saveImgFolderPath, f'{imgPath.stem}_DA{batchNum}.jpg')
        saveXmlFilePath =  os.path.join(saveXmlFolderPath, f'{imgPath.stem}_DA{batchNum}.xml')

        # img load
        image = cv2.imread(item)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    # 確認是否需要cvtColor

        # xml load
        tree = ET.parse(os.path.join(xmlFolderPath, f'{imgPath.stem}.xml'))
        root = tree.getroot()
        obj = root.findall('object')



        # read bounding box info
        BoundingBoxList =  []
        for item in obj:
            bndbox = item.find('bndbox')
            BoundingBoxList.append(ia.BoundingBox(
                x1=int(bndbox[0].text), 
                y1=int(bndbox[1].text), 
                x2=int(bndbox[2].text), 
                y2=int(bndbox[3].text)))
        # bunding box info setting 
        bbs = ia.BoundingBoxesOnImage(BoundingBoxList, shape=image.shape)


        # Augmentation parameter setting
        if seq==None:
            # seq = iaa.Sequential([
                # Augmentation #1 (扭曲、雜訊) CoarseDropout PiecewiseAffine
                # iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5),
                # iaa.PiecewiseAffine(scale=(0.05, 0.1))

                # Augmentation #3 (顏色)color
                # iaa.ChangeColorTemperature((1100, 10000)),
                # iaa.AddToHueAndSaturation((-50, 50), per_channel=True),

                # Augmentation #4 (Conv) convolutional
                # iaa.SomeOf(1, [
                #     iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
                #     iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),
                #     iaa.EdgeDetect(alpha=(0.0, 1.0)),
                #     iaa.DirectedEdgeDetect(alpha=(0.0, 1.0), direction=(0.0, 1.0))
                #      ]),

                # Augmentation #5 (像素層級的改變光影)
                # iaa.MultiplyElementwise((0.5, 1.5), per_channel=0.5),
                
                # Augmentation #6 (縮放) scale
                # iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)}),


            #     iaa.Affine(rotate=(-45, 45)),
            #     iaa.Flipud(0.3),
            #     iaa.Fliplr(0.3),
            #     iaa.Multiply((0.5, 1.5)) # 改變亮度, 不影響bounding box
            #  ])

            # Augmentation #2 (MixUp、頻率域雜訊) BlendAlphaFrequencyNoise
            seq = iaa.BlendAlphaFrequencyNoise(
                    exponent=(-2.5, -1.0),
                    foreground=iaa.Affine(
                        rotate=(-45, 45),
                        translate_px={"x": (-4, 4), "y": (-4, 4)}
                    ),
                    background=iaa.AddToHueAndSaturation((-180, 180)),
                    per_channel=True
            )


        # Augmentation image / augment_bounding_boxes
        seq_det = seq.to_deterministic()
        image_aug = seq_det.augment_images([image])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]



        # update bounding box info
        for i in range(len(obj)):
            bndbox = obj[i].find('bndbox')
            bndbox[0].text = str(int(bbs_aug.bounding_boxes[i].x1))
            bndbox[1].text = str(int(bbs_aug.bounding_boxes[i].y1))
            bndbox[2].text = str(int(bbs_aug.bounding_boxes[i].x2))
            bndbox[3].text = str(int(bbs_aug.bounding_boxes[i].y2))
        
        # save augmentation xml 
        tree.write(saveXmlFilePath)

        # save augmentation image
        print('save image: ',saveImgFilePath)
        cv2.imwrite(saveImgFilePath,image_aug)
        
        # imshow
        image_after = bbs_aug.draw_on_image(image_aug, size=5)
        cv2.imshow('windows', image_after)
        cv2.waitKey(1)
 


if __name__ =="__main__":

    # Augmentation #7~finish
    seq = ImaugFunctions(all_some_one='one', 
    use_arithmetic_num=999, 
    use_blend_num=999, 
    use_blur_num=999,  
    use_color_num=999, 
    use_contrast_num=999, 
    use_flip_num=999,
    use_geometric_num=999,
    use_pooling_num=999,
    use_weather_num=999)

    for item in range(batchRunNum):
        DataAugmentationFlow(seq.__call__(1))
        batchNum = batchNum + 1
