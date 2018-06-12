from torch.utils.data import Dataset,DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.image as mpimg

class FacialKeypointsDataset(Dataset):

    def __init__(self,csv_file,root_dir,transform=None):
        """
        Args:
             csv_file (string)
             root_dir
             transform
        """
        self.key_pts_frame=pd.read_csv(csv_file)
        self.root_dir=root_dir
        self.transform=transform

    def __len__(self):
        return len(self.key_pts_frame)


    def __getitem__(self,idx):
        image_name=os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx,0])
        image=mpimg.imread(image_name)

        if (image.shape[2]==4):
            image=image[:,:,:3]
        key_pts=self.key_pts_frame.iloc[idx,1:].as_matrix()
        key_pts=key_pts.astype('float').reshape(-1,2)
        sample={'image':image,'keypoints':key_pts}

        if self.transform:
            sample=self.transform(sample)
        return sample

if __name__=='__main__':

    def show_keypoints(image,key_pts):
        """Show image with keypoints"""
        plt.imshow(image)
        plt.scatter(key_pts[:,0],key_pts[:,1],s=20, marker='.',c='m')
        plt.show()
    
    face_dataset=FacialKeypointsDataset(csv_file='./data/training_frames_keypoints.csv',
                                       root_dir='./data/training/')

    print('Length of dataset:',len(face_dataset))


    num_to_display=3

    for i in range(num_to_display):

        fig=plt.figure(figsize=(20,10))

        rand_i=np.random.randint(0,len(face_dataset))
        sample=face_dataset[rand_i]

        print(i,sample['image'].shape,sample['keypoints'].shape)

        ax=plt.subplot(1,num_to_display,i+1)
        ax.set_title('Sample #{}'.format(i))

        show_keypoints(sample['image'],sample['keypoints'])
