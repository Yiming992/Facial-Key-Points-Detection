from torchvision import transforms,utils
import torch


class Normalize():

    def __call__(self,sample):

        image,key_pts=sample['image'],sample['keypoints']

        image_copy=np.copy(image)
        key_pts_copy=np.copy(key_pts)

        image_copy=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

        image_copy=image_copy/255.0

        key_pts_copy=(key_pts_copy-100)/50.0

        return {'image':image_copy,'keypoints':key_pts_copy}

class Rescale():

    def __init__(self,output_size):

        assert isinstance(output_size,(int,tuple))
        self.output_size=output_size

    def __call__(self,sample):

        image,key_pts=sample['image'],sample['keypoints']

        h,w=image.shape[:2]

        if isinstance(self.output_size,int):
            if h>w:

                new_h,new_w=self.output_size*h/w,self.output_size
            else:

                new_h,new_w=self,output_size,self.output_size*w/h
        else:
            new_h,new_w=int(new_h),int(new_w)

        new_h,new_w=int(new_h),int(new_w)

        img=cv2.resize(image,(new_w,new_h))

        key_pts=key_pts*[new_w/w,new_h/h]

        return {'image':img,'keypoints':key_pts}

class RandomCrop():

    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))

        if isinstance(output_size,int):
            self.output_size=(output_size,output_size)
        else:
            assert len(output_size)==2
            self.output_size=output_size

    def __call__(self,sample):
        image,key_pts=sample['image'],sample['keypoints']

        h,w=image.shape[:2]
        new_h,new_w=self.output_size
        
        top=np.random.randint(0,h-new_h)
        left=np.random.randint(0,w-new_w)

        image=image[top:top+new_h,left:left+new_w]

        key_pts=key_pts-[left,top]

        return {'image':image,'keypoints':key_pts}




class ToTensor():

    def __call__(self,sample):
        image,key_pts=sample['image'],sample['keypoints']

        if (len(image.shape)==2):

            image=image.reshape(image.shape[0],image.shape[1],1)

            image=image.transpose((2,0,1))

            return {'image':torch.from_numpy(image),'keypoints':torch.from_numpy(key_pts)}



if __name__=='__main__':

    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    def show_keypoints(image,key_pts):
        """Show image with keypoints"""
        plt.imshow(image)
        plt.scatter(key_pts[:,0],key_pts[:,1],s=20, marker='.',c='m')

    # Test above specified transforms
    from Dataset import FacialKeypointsDataset
    # face_dataset=FacialKeypointsDataset(csv_file='./data/training_frames_keypoints.csv',
    #                                    root_dir='./data/training/')

    # rescale=Rescale(100)
    # crop=RandomCrop(50)
    # composed=transforms.Compose([Rescale(250),
    #                             RandomCrop(224)])

    # test_num=500
    # sample=face_dataset[test_num]

    # fig=plt.figure()

    # for i,tx in enumerate([rescale,crop,composed]):
    #     transformed_sample=tx(sample)

    #     ax=plt.subplot(1,3,i+1)
    #     plt.tight_layout()
    #     ax.set_title(type(tx).__name__)
    #     show_keypoints(transformed_sample['image'],transformed_sample['keypoints'])

    # plt.show()

    data_transform=transforms.Compose([Rescale(250),RandomCrop(224),Normalize(),ToTensor()])

    transformed_dataset=FacialKeypointsDataset(csv_file='./data/training_frames_keypoints.csv',
                                               root_dir='./data/training/',
                                               transform=data_transform)

    print('Number of images: ', len(transformed_dataset))

    for i in range(5):
        sample=transformed_dataset[i]
        print(i,sample['image'].size(),sample['keypoints'].size())




                 



