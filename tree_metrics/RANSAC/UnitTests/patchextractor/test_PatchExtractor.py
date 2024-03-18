import unittest
from RANSAC.Algorithm import ImagePatchExtractor
from RANSAC.Common import PatchInfo,PatchResults

import skimage
import os


class Test_test_PatchExtractor(unittest.TestCase):
    def read_test_image(self,filename,gray=True):
        folder_script=os.path.dirname(__file__)
        file_noisy_line=os.path.join(folder_script,"./data/",filename)
        np_image=skimage.io.imread(file_noisy_line,as_gray=gray)
        return np_image,file_noisy_line

    def test_construction(self):
        stride=11
        size=20
        x=ImagePatchExtractor(None,size,stride)
        self.assertEquals(x.stride,stride)
        self.assertEquals(x.size,size)
        pass

    def test_W_7_H_10_filter_4_stride_2(self):
        np_image,absolute_path=self.read_test_image("SimplePatchTest_W=7_H=10.png")
        filter_size=4
        filter_stride=2
        xtractor=ImagePatchExtractor(np_image,filter_size,filter_stride)
        patch_results:PatchResults=xtractor.extract_patches();
        all_patches=patch_results.all_patches
        self.assertEquals(len(all_patches),12) #4 X 4 patches
        arr_of_indices=patch_results.patch_indices
        #Test the array of patches
        self.assertEquals(arr_of_indices.shape[0],4)
        self.assertEquals(arr_of_indices.shape[1],3)
        #examine each and every patch
        for y in range(0,arr_of_indices.shape[0]):
            for x in range(0,arr_of_indices.shape[1]):
                patch=patch_results.get_patch_xy(x,y)
                patch_width=patch.image.shape[1]
                patch_height=patch.image.shape[0]
                self.assertEquals(patch_width,filter_size) 
                self.assertEquals(patch_height,filter_size) 
        
    def test_W_4_H_4_filter_4_stride_0(self):
        np_image,absolute_path=self.read_test_image("SimplePatchTest_W=4_H=4.png")
        filter_size=4
        filter_stride=0
        xtractor=ImagePatchExtractor(np_image,filter_size,filter_stride)
        patch_results:PatchResults=xtractor.extract_patches();
        #Just 1 patch because stride=0 and filter matches image dimensions
        all_patches=patch_results.all_patches
        self.assertEquals(len(all_patches),1) 
        
        #Test the array of patches - just 1 cell because only 1 patch
        arr_of_indices=patch_results.patch_indices
        self.assertEquals(arr_of_indices.shape[0],1)
        self.assertEquals(arr_of_indices.shape[1],1)

        #Inspect the contents of the only patch
        patch_0_0_info=patch_results.get_patch_xy(0,0)
        
        patch_0_0=patch_0_0_info.image
        patch_width=patch_0_0.shape[1]
        patch_height=patch_0_0.shape[0]
        self.assertEquals(patch_width,filter_size) 
        self.assertEquals(patch_height,filter_size) 

        black=0
        white=1
        self.assertEquals(patch_0_0[0,0],black)
        self.assertEquals(patch_0_0[1,0],black)
        self.assertEquals(patch_0_0[2,0],black)
        self.assertEquals(patch_0_0[3,0],black)

        self.assertEquals(patch_0_0[3,0],black)
        self.assertEquals(patch_0_0[3,1],black)
        self.assertEquals(patch_0_0[3,2],black)
        self.assertEquals(patch_0_0[3,3],black)

        self.assertEquals(patch_0_0[0,1],white)
        self.assertEquals(patch_0_0[0,2],white)
        self.assertEquals(patch_0_0[0,3],white)
        pass

    def test_W_5_H_5_filter_4_stride_0(self):
        np_image,absolute_path=self.read_test_image("SimplePatchTest_W=5_H=5.png")
        self.assertEquals(np_image.shape[0],5)
        self.assertEquals(np_image.shape[1],5)
        filter_size=4
        filter_stride=2
        xtractor=ImagePatchExtractor(np_image,filter_size,filter_stride)
        patch_results:PatchResults=xtractor.extract_patches();

        arr_of_indices=patch_results.patch_indices
        all_patches=patch_results.all_patches

        #Just 2X2 patches because stride=2 and filter=4, image width=5 ,therefore 2 movements along each axis
        self.assertEquals(len(all_patches),4) 
        self.assertEquals(arr_of_indices.shape[0],2)
        self.assertEquals(arr_of_indices.shape[1],2)

        #Examine the size of each patch
        for y in range(0,arr_of_indices.shape[0]):
            for x in range(0,arr_of_indices.shape[1]):
                patch=patch_results.get_patch_xy(y,x)
                patch_width=patch.image.shape[1]
                patch_height=patch.image.shape[0]
                self.assertEquals(patch_width,filter_size) 
                self.assertEquals(patch_height,filter_size) 
        
        

        #Inspect the contents of the only patch

        #top-left
        patch_0_0_info=patch_results.get_patch_xy(0,0)
        patch_0_0=patch_0_0_info.image
        self.assertEquals(patch_0_0_info.topleft.X,0) 
        self.assertEquals(patch_0_0_info.topleft.Y,0) 
        self.assertEquals(patch_0_0_info.bottomright.X,3) 
        self.assertEquals(patch_0_0_info.bottomright.Y,3) 

        #top-right
        patch_1_0_info=patch_results.get_patch_xy(1,0)
        patch_1_0=patch_1_0_info.image
        self.assertEquals(patch_1_0_info.topleft.X,1) 
        self.assertEquals(patch_1_0_info.topleft.Y,0) 
        self.assertEquals(patch_1_0_info.bottomright.X,4) 
        self.assertEquals(patch_1_0_info.bottomright.Y,3) 

        #bottom-left
        patch_0_1_info=patch_results.get_patch_xy(0,1)
        patch_0_1=patch_0_1_info.image
        self.assertEquals(patch_0_1_info.topleft.X,0)
        self.assertEquals(patch_0_1_info.topleft.Y,1) 
        self.assertEquals(patch_0_1_info.bottomright.X,3)
        self.assertEquals(patch_0_1_info.bottomright.Y,4) 

        #bottom-right
        patch_1_1_info=patch_results.get_patch_xy(1,1)
        patch_1_1=patch_1_1_info.image
        self.assertEquals(patch_1_1_info.topleft.X,1)
        self.assertEquals(patch_1_1_info.topleft.Y,1) 
        self.assertEquals(patch_1_1_info.bottomright.X,4)
        self.assertEquals(patch_1_1_info.bottomright.Y,4) 


if __name__ == '__main__':
    unittest.main()




    #self.fail("Not implemented")