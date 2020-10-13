#include <algorithm>
#include <iostream>
#include "caffe/common.hpp"
#include "caffe/util/im2col.hpp"
#include "math.h"

#include <stdio.h>

namespace caffe {

template <typename Dtype>
__device__ Dtype deformable_im2col_bilinear(const Dtype* bottom_data, const int data_width, const int data_height, float h, float w, int index, int test_index) {

    int h_low = floor(h);
    int w_low = floor(w);

    int h_high = h_low + 1;
    int w_high = w_low + 1;

    if(h_low<0)
        h_low = h_low + data_height;
    if(w_low<0)
        w_low = w_low + data_width;
   if(h_high>=data_height)
	h_high= h_high - data_height;
   if(w_high>=data_width)
        w_high = w_high - data_width;

    Dtype u = h - h_low;
    Dtype v = w - w_low;
    Dtype v1 = bottom_data[h_low * data_width + w_low];
    Dtype v2 = bottom_data[h_low * data_width + w_high];
    Dtype v3 = bottom_data[h_high * data_width + w_low];
    Dtype v4 = bottom_data[h_high * data_width + w_high];

    Dtype val = (1-u)*(1-v)*v1 + (1-u)*v*v2 + u*(1-v)*v3 + u*v*v4;
    if(!val)
        val=0;
    return val;
}

template <typename Dtype>
__device__ Dtype deformable_im2col_floor(const Dtype* bottom_data, const int data_width, const int data_height, float h, float w, int index, int test_index) {

    int h_low = floor(h);
    int w_low = floor(w);

    if(h_low<0)
        h_low = h_low + data_height;
    if(w_low<0)
        w_low = w_low + data_width;

    Dtype val = bottom_data[h_low * data_width + w_low];
    return val;
}

template <typename Dtype>
__device__ Dtype deformable_im2col_nearest(const Dtype* bottom_data, const int data_width, const int data_height, float h, float w, int index, int test_index) {

    int h_low = round(h);
    int w_low = round(w);

    if(h_low<0)
        h_low = h_low + data_height;
    if(w_low<0)
        w_low = w_low + data_width;
    if(h_low>=data_height)
	h_low= h_low - data_height;
    if(w_low>=data_width)
        w_low = w_low - data_width;

    Dtype val = bottom_data[h_low * data_width + w_low];
    return val;
}

template <typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    const int h_index = index / width_col;
    const int h_col = h_index % height_col;
    const int w_col = index % width_col;
    const int c_im = h_index / height_col;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;
    Dtype* data_col_ptr = data_col;

    int test_index = width_col/2+width_col*height_col/2;//+width_col*height_col;
    int r_scale = 1;

    int h_offset_decal = kernel_h/2;
    int w_offset_decal = kernel_w/2;
    int img_r = h_offset+h_offset_decal;
    int img_c = w_offset+w_offset_decal;

    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    const Dtype* data_im_ptr = data_im;
    //data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    data_im_ptr += c_im * height * width;

	/*
/////// HAOZHOU //////////////////////
    float phi = -(img_r-height/2)*M_PI/height;//-((img_r+0.5)/height*M_PI - M_PI/2);
    float theta = (img_c-width/2)*2*M_PI/width;//(img_c+0.5)/width*2*M_PI-M_PI;
    float delta_phi = M_PI/height;
    float delta_theta = 2*M_PI/width;

    //The coodinates of sampling grid elements are saved in sphere_conv
    float sphere_conv[7][7][2];
    if(kernel_h==3){
    sphere_conv[0][0][0] = -tan(delta_theta);
    sphere_conv[0][0][1] = tan(delta_phi)/cos(delta_theta);    
    sphere_conv[0][1][0] = 0;
    sphere_conv[0][1][1] = tan(delta_phi);
    sphere_conv[0][2][0] = tan(delta_theta);
    sphere_conv[0][2][1] = tan(delta_phi)/cos(delta_theta);    
    sphere_conv[1][0][0] = -tan(delta_theta);
    sphere_conv[1][0][1] = 0;
    sphere_conv[1][1][0] = 1;
    sphere_conv[1][1][1] = 1;
    sphere_conv[1][2][0] = tan(delta_theta);
    sphere_conv[1][2][1] = 0;
    sphere_conv[2][0][0] = -tan(delta_theta);
    sphere_conv[2][0][1] = -tan(delta_phi)/cos(delta_theta);    
    sphere_conv[2][1][0] = 0;
    sphere_conv[2][1][1] = -tan(delta_phi);
    sphere_conv[2][2][0] = tan(delta_theta);
    sphere_conv[2][2][1] = -tan(delta_phi)/cos(delta_theta);
    }
    else if(kernel_h==5){

    sphere_conv[0][0][0] = -tan(2*delta_theta);
    sphere_conv[0][0][1] = 1/cos(2*delta_theta)*tan(2*delta_phi);
    sphere_conv[0][1][0] = -tan(delta_theta); 
    sphere_conv[0][1][1] = 1/cos(2*delta_theta)*tan(2*delta_phi);
    sphere_conv[0][2][0] = 0;
    sphere_conv[0][2][1] = 1/cos(2*delta_theta)*tan(2*delta_phi);
    sphere_conv[0][3][0] = tan(delta_theta);
    sphere_conv[0][3][1] = 1/cos(2*delta_theta)*tan(2*delta_phi);
    sphere_conv[0][4][0] = tan(2*delta_theta);
    sphere_conv[0][4][1] = 1/cos(2*delta_theta)*tan(2*delta_phi);

    sphere_conv[1][0][0] = -tan(2*delta_theta);
    sphere_conv[1][0][1] = 1/cos(delta_theta)*tan(delta_phi);
    sphere_conv[1][1][0] = -tan(delta_theta);
    sphere_conv[1][1][1] = 1/cos(delta_theta)*tan(delta_phi);
    sphere_conv[1][2][0] = 0;
    sphere_conv[1][2][1] = 1/cos(delta_theta)*tan(delta_phi);
    sphere_conv[1][3][0] = tan(delta_theta);
    sphere_conv[1][3][1] = 1/cos(delta_theta)*tan(delta_phi);
    sphere_conv[1][4][0] = tan(2*delta_theta);
    sphere_conv[1][4][1] = 1/cos(delta_theta)*tan(delta_phi);

    sphere_conv[2][0][0] = -tan(2*delta_theta);
    sphere_conv[2][0][1] = 0;
    sphere_conv[2][1][0] = -tan(delta_theta);
    sphere_conv[2][1][1] = 0;
    sphere_conv[2][2][0] = 1;
    sphere_conv[2][2][1] = 1;
    sphere_conv[2][3][0] = tan(delta_theta);
    sphere_conv[2][3][1] = 0;
    sphere_conv[2][4][0] = tan(2*delta_theta);
    sphere_conv[2][4][1] = 0;

    sphere_conv[3][0][0] = -tan(2*delta_theta);
    sphere_conv[3][0][1] = -1/cos(delta_theta)*tan(delta_phi);
    sphere_conv[3][1][0] = -tan(delta_theta);
    sphere_conv[3][1][1] = -1/cos(delta_theta)*tan(delta_phi);
    sphere_conv[3][2][0] = 0;
    sphere_conv[3][2][1] = -1/cos(delta_theta)*tan(delta_phi);
    sphere_conv[3][3][0] = tan(delta_theta);
    sphere_conv[3][3][1] = -1/cos(delta_theta)*tan(delta_phi);
    sphere_conv[3][4][0] = tan(2*delta_theta);
    sphere_conv[3][4][1] = -1/cos(delta_theta)*tan(delta_phi);

    sphere_conv[4][0][0] = -tan(2*delta_theta);
    sphere_conv[4][0][1] = -1/cos(2*delta_theta)*tan(2*delta_phi);
    sphere_conv[4][1][0] = -tan(delta_theta);
    sphere_conv[4][1][1] = -1/cos(2*delta_theta)*tan(2*delta_phi);
    sphere_conv[4][2][0] = 0;
    sphere_conv[4][2][1] = -1/cos(2*delta_theta)*tan(2*delta_phi);
    sphere_conv[4][3][0] = tan(delta_theta);
    sphere_conv[4][3][1] = -1/cos(2*delta_theta)*tan(2*delta_phi);
    sphere_conv[4][4][0] = tan(2*delta_theta);
    sphere_conv[4][4][1] = -1/cos(2*delta_theta)*tan(2*delta_phi);

    }
    else if(kernel_h==7){
    sphere_conv[0][0][0] = -tan(3*delta_theta);
    sphere_conv[0][0][1] = 1/cos(3*delta_theta)*tan(3*delta_phi);
    sphere_conv[0][1][0] = -tan(2*delta_theta);
    sphere_conv[0][1][1] = 1/cos(3*delta_theta)*tan(3*delta_phi);
    sphere_conv[0][2][0] = -tan(delta_theta);
    sphere_conv[0][2][1] = 1/cos(3*delta_theta)*tan(3*delta_phi);
    sphere_conv[0][3][0] = 0;
    sphere_conv[0][3][1] = 1/cos(3*delta_theta)*tan(3*delta_phi);
    sphere_conv[0][4][0] = tan(delta_theta);
    sphere_conv[0][4][1] = 1/cos(3*delta_theta)*tan(3*delta_phi);
    sphere_conv[0][5][0] = tan(2*delta_theta);
    sphere_conv[0][5][1] = 1/cos(3*delta_theta)*tan(3*delta_phi);
    sphere_conv[0][6][0] = tan(3*delta_theta);
    sphere_conv[0][6][1] = 1/cos(3*delta_theta)*tan(3*delta_phi);

    sphere_conv[1][0][0] = -tan(3*delta_theta);
    sphere_conv[1][0][1] = 1/cos(2*delta_theta)*tan(2*delta_phi);
    sphere_conv[1][1][0] = -tan(2*delta_theta);
    sphere_conv[1][1][1] = 1/cos(2*delta_theta)*tan(2*delta_phi);
    sphere_conv[1][2][0] = -tan(delta_theta);
    sphere_conv[1][2][1] = 1/cos(2*delta_theta)*tan(2*delta_phi);
    sphere_conv[1][3][0] = 0;
    sphere_conv[1][3][1] = 1/cos(2*delta_theta)*tan(2*delta_phi);
    sphere_conv[1][4][0] = tan(delta_theta);
    sphere_conv[1][4][1] = 1/cos(2*delta_theta)*tan(2*delta_phi);
    sphere_conv[1][5][0] = tan(2*delta_theta);
    sphere_conv[1][5][1] = 1/cos(2*delta_theta)*tan(2*delta_phi);
    sphere_conv[1][6][0] = tan(3*delta_theta);
    sphere_conv[1][6][1] = 1/cos(2*delta_theta)*tan(2*delta_phi);

    sphere_conv[2][0][0] = -tan(3*delta_theta);
    sphere_conv[2][0][1] = 1/cos(delta_theta)*tan(delta_phi);
    sphere_conv[2][1][0] = -tan(2*delta_theta);
    sphere_conv[2][1][1] = 1/cos(delta_theta)*tan(delta_phi);
    sphere_conv[2][2][0] = -tan(delta_theta);
    sphere_conv[2][2][1] = 1/cos(delta_theta)*tan(delta_phi);
    sphere_conv[2][3][0] = 0;
    sphere_conv[2][3][1] = 1/cos(delta_theta)*tan(delta_phi);
    sphere_conv[2][4][0] = tan(delta_theta);
    sphere_conv[2][4][1] = 1/cos(delta_theta)*tan(delta_phi);
    sphere_conv[2][5][0] = tan(2*delta_theta);
    sphere_conv[2][5][1] = 1/cos(delta_theta)*tan(delta_phi);
    sphere_conv[2][6][0] = tan(3*delta_theta);
    sphere_conv[2][6][1] = 1/cos(delta_theta)*tan(delta_phi);

    sphere_conv[3][0][0] = -tan(3*delta_theta);
    sphere_conv[3][0][1] = 0;
    sphere_conv[3][1][0] = -tan(2*delta_theta);
    sphere_conv[3][1][1] = 0;
    sphere_conv[3][2][0] = -tan(delta_theta);
    sphere_conv[3][2][1] = 0;
    sphere_conv[3][3][0] = 1;
    sphere_conv[3][3][1] = 1;
    sphere_conv[3][4][0] = tan(delta_theta);
    sphere_conv[3][4][1] = 0;
    sphere_conv[3][5][0] = tan(2*delta_theta);
    sphere_conv[3][5][1] = 0;
    sphere_conv[3][6][0] = tan(3*delta_theta);
    sphere_conv[3][6][1] = 0;

    sphere_conv[4][0][0] = -tan(3*delta_theta);
    sphere_conv[4][0][1] = -1/cos(delta_theta)*tan(delta_phi);
    sphere_conv[4][1][0] = -tan(2*delta_theta);
    sphere_conv[4][1][1] = -1/cos(delta_theta)*tan(delta_phi);
    sphere_conv[4][2][0] = -tan(delta_theta);
    sphere_conv[4][2][1] = -1/cos(delta_theta)*tan(delta_phi);
    sphere_conv[4][3][0] = 0;
    sphere_conv[4][3][1] = -1/cos(delta_theta)*tan(delta_phi);
    sphere_conv[4][4][0] = tan(delta_theta);
    sphere_conv[4][4][1] = -1/cos(delta_theta)*tan(delta_phi);
    sphere_conv[4][5][0] = tan(2*delta_theta);
    sphere_conv[4][5][1] = -1/cos(delta_theta)*tan(delta_phi);
    sphere_conv[4][6][0] = tan(3*delta_theta);
    sphere_conv[4][6][1] = -1/cos(delta_theta)*tan(delta_phi);

    sphere_conv[5][0][0] = -tan(3*delta_theta);
    sphere_conv[5][0][1] = -1/cos(2*delta_theta)*tan(2*delta_phi);
    sphere_conv[5][1][0] = -tan(2*delta_theta);
    sphere_conv[5][1][1] = -1/cos(2*delta_theta)*tan(2*delta_phi);
    sphere_conv[5][2][0] = -tan(delta_theta);
    sphere_conv[5][2][1] = -1/cos(2*delta_theta)*tan(2*delta_phi);
    sphere_conv[5][3][0] = 0;
    sphere_conv[5][3][1] = -1/cos(2*delta_theta)*tan(2*delta_phi);
    sphere_conv[5][4][0] = tan(delta_theta);
    sphere_conv[5][4][1] = -1/cos(2*delta_theta)*tan(2*delta_phi);
    sphere_conv[5][5][0] = tan(2*delta_theta);
    sphere_conv[5][5][1] = -1/cos(2*delta_theta)*tan(2*delta_phi);
    sphere_conv[5][6][0] = tan(3*delta_theta);
    sphere_conv[5][6][1] = -1/cos(2*delta_theta)*tan(2*delta_phi);

    sphere_conv[6][0][0] = -tan(3*delta_theta);
    sphere_conv[6][0][1] = -1/cos(3*delta_theta)*tan(3*delta_phi);
    sphere_conv[6][1][0] = -tan(2*delta_theta);
    sphere_conv[6][1][1] = -1/cos(3*delta_theta)*tan(3*delta_phi);
    sphere_conv[6][2][0] = -tan(delta_theta);
    sphere_conv[6][2][1] = -1/cos(3*delta_theta)*tan(3*delta_phi);
    sphere_conv[6][3][0] = 0;
    sphere_conv[6][3][1] = -1/cos(3*delta_theta)*tan(3*delta_phi);
    sphere_conv[6][4][0] = tan(delta_theta);
    sphere_conv[6][4][1] = -1/cos(3*delta_theta)*tan(3*delta_phi);
    sphere_conv[6][5][0] = tan(2*delta_theta);
    sphere_conv[6][5][1] = -1/cos(3*delta_theta)*tan(3*delta_phi);
    sphere_conv[6][6][0] = tan(3*delta_theta);
    sphere_conv[6][6][1] = -1/cos(3*delta_theta)*tan(3*delta_phi);

    }

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h_im = h_offset + i * dilation_h;
        int w_im = w_offset + j * dilation_w;
        float x = sphere_conv[i][j][0];
        float y = sphere_conv[i][j][1];
        float rho = sqrt(x*x+y*y);
        float v = atan(rho);
        float new_phi = asin(cos(v)*sin(phi) + y*sin(v)*cos(phi)/rho);
        float new_theta = theta + atan(x*sin(v) / (rho*cos(phi)*cos(v) - y*sin(phi)*sin(v)));
	float new_r = (-new_phi/M_PI+0.5)*height;
	//(-new_phi+M_PI/2)*height/M_PI - 0.5;
	float new_c = (new_theta/2/M_PI+0.5)*width;
	//(new_theta+M_PI)*width/2/M_PI - 0.5;

	while(new_r<0)
            new_r = new_r + height;
        while(new_c<0)
            new_c = new_c + width;
        while(new_r>=height)
            new_r = new_r - height;
        while(new_c>=width)
            new_c = new_c - width;
	if(kernel_h==3){
        if(i==1 && j==1){
            new_r = img_r;
            new_c = img_c;
        }
	}
	else if(kernel_h==5){
	        if(i==2 && j==2){
            new_r = img_r;
            new_c = img_c;
	}
	}
        else if(kernel_h==7){ 
                if(i==3 && j==3){
            new_r = img_r;
            new_c = img_c;
        }
	}
	if(index==test_index && width==384){
		printf("HAOZHOU\n");
		printf("i , j: %d %d\n",i,j);
		printf("new_phi: %f\n",new_phi);
		printf("new_theta: %f\n",new_theta);
		printf("new_r: %f\n",new_r);
		printf("new_c: %f\n",new_c);
	 }
        Dtype val = static_cast<Dtype>(0);
	if(new_r>=0 && new_r<height && new_c>=0 && new_c<width){
		val = deformable_im2col_bilinear(data_im_ptr, width, height, new_r, new_c, index, test_index);
		if(index==test_index && width==384){
			printf("val = %f\n",val);
	    		printf("\n\n");
		}
	}

	}
	}
	*/

/////// CLARA //////////////////////
    int r_w = kernel_w;
    int r_h = kernel_h;
    float alpha = 2*M_PI/width;
    float d=1/(2*tan(alpha/2));
    
    float phi0 = (img_c-width/2)*2*M_PI/width;
    float omega0 = -(img_r-height/2)*M_PI/height;
  
    /*
    // CAFFE STANDARD CONVOLUTION OPERATION 
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    const Dtype* data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h_im = h_offset + i * dilation_h;
        int w_im = w_offset + j * dilation_w;
        *data_col_ptr =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
            data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
        data_col_ptr += height_col * width_col;
      }
    }

    // CAFFE STANDARD CONVOLUTION OPERATION MODIFIED FOR OUTPUT
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h_im = h_offset + i * dilation_h;
        int w_im = w_offset + j * dilation_w;
	if(index==test_index && width==768){
		printf("index = %d\n",index);	
		printf("dilation_h, dilation_w = %d %d\n",dilation_h,dilation_w);	
		printf("h_im, w_im = %d %d\n",h_im,w_im);
		printf("img_r, img_c = %d %d\n",img_r,img_c);
		printf("data_col_ptr += %d\n",(c_col * height_col + h_col) * width_col + w_col);
		printf("data_im_ptr += %d \n",(c_im * height + h_offset) * width + w_offset);

		printf("val_coord, val_origin = %d %f\n",i * dilation_h * width + j * dilation_w,data_im_ptr[i * dilation_h * width + j * dilation_w]);
		printf("val_decal = %d\n",height_col * width_col);
    		printf("\n\n");
	}
      }
    }

	*/

    // CAFFE NEW CONVOLUTION OPERATION TO ADAPT TO OMNI
    for (int j = (r_h-1)/2; j > -(r_h-1)/2-1; --j) {
      for (int i = -(r_w-1)/2; i < (r_w-1)/2+1; ++i) {
                float X_temp = cos(phi0)*r_scale*i+sin(phi0)*(sin(-omega0)*r_scale*j+cos(-omega0)*d);
                float Y_temp = cos(-omega0)*r_scale*j-sin(-omega0)*d;
                float Z_temp = -sin(phi0)*r_scale*i+cos(phi0)*(sin(-omega0)*r_scale*j+cos(-omega0)*d);

		float phi_temp = atan2(X_temp,Z_temp);
		float omega_temp = asin(Y_temp/sqrt((r_scale*i)*(r_scale*i)+(r_scale*j)*(r_scale*j)+d*d));

		float u_temp = (phi_temp/(2*M_PI)+0.5)*width;
		float v_temp = (-omega_temp/M_PI+0.5)*height;

		float new_c = u_temp;
		float new_r = v_temp;

		while(new_r<0)
            		new_r = new_r + height;
        	while(new_c<0)
            		new_c = new_c + width;
		while(new_r>=height)
			new_r = new_r - height;
		while(new_c>=width)
			new_c = new_c - width;

		Dtype val = static_cast<Dtype>(0);
		if(new_r>=0 && new_r<height && new_c>=0 && new_c<width){
			val = deformable_im2col_bilinear(data_im_ptr, width, height, new_r, new_c, index, test_index);
		}

		/*
		if(index==test_index && width==384){
			printf("CLARA\n");
			printf("i , j: %d %d\n",i,j);
			printf("new_phi: %f\n",omega_temp);
			printf("new_theta: %f\n",phi_temp);
			printf("new_r: %f\n",new_r);
			printf("new_c: %f\n",new_c);
			printf("val = %f\n",val);
		    	printf("\n\n");
	    	}
		*/

	*data_col_ptr = val;
	 
        data_col_ptr += height_col * width_col;
      }
    }

  }
}

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
      width_col, data_col);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void im2col_gpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, float* data_col);
template void im2col_gpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, double* data_col);

template <typename Dtype, int num_axes>
__global__ void im2col_nd_gpu_kernel(const int n, const Dtype* data_im,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_col) {
  int d_temp[num_axes];  // NOLINT(runtime/arrays)
  int d_iter[num_axes];  // NOLINT(runtime/arrays)

  __shared__ int shared_dilation[num_axes];
  __shared__ int shared_kernel_shape[num_axes];
  __shared__ int shared_pad[num_axes];
  __shared__ int shared_stride[num_axes];
  __shared__ int shared_col_shape[num_axes + 1];
  __shared__ int shared_im_shape[num_axes + 1];

  if (threadIdx.x < num_axes) {
    shared_dilation[threadIdx.x] = dilation[threadIdx.x];
    shared_kernel_shape[threadIdx.x] = kernel_shape[threadIdx.x];
    shared_pad[threadIdx.x] = pad[threadIdx.x];
    shared_stride[threadIdx.x] = stride[threadIdx.x];
  }
  if (threadIdx.x < num_axes + 1) {
    shared_col_shape[threadIdx.x] = col_shape[threadIdx.x];
    shared_im_shape[threadIdx.x] = im_shape[threadIdx.x];
  }
  __syncthreads();

  int i;
  CUDA_KERNEL_LOOP(index, n) {
    // Initialize channel_in, computed in the loop below, with intermediate
    // computations used to compute the spatial indices.
    int channel_in = index;
    int channel_out = 1;
    for (i = num_axes - 1; i >= 0; --i) {
      d_temp[i] = channel_in % shared_col_shape[i + 1];
      channel_in /= shared_col_shape[i + 1];
      channel_out *= shared_kernel_shape[i];
    }
    channel_out *= channel_in;
    int data_col_inc = 1;
    for (i = 0; i < num_axes; ++i) {
      channel_out *= shared_col_shape[i + 1];
      channel_out += d_temp[i];
      d_temp[i] = d_temp[i] * shared_stride[i] - shared_pad[i];
      channel_in *= shared_im_shape[i + 1];
      channel_in += d_temp[i];
      data_col_inc *= shared_col_shape[i + 1];
      d_iter[i] = 0;
    }
    Dtype* data_col_ptr = data_col + channel_out;
    const Dtype* data_im_ptr = data_im + channel_in;
    bool incremented;
    do {
      bool in_range = true;
      for (i = 0; i < num_axes; ++i) {
        const int d_iter_im = d_iter[i] * shared_dilation[i] + d_temp[i];
        in_range &= d_iter_im >= 0 && d_iter_im < shared_im_shape[i + 1];
        if (!in_range) { break; }
      }
      if (in_range) {
        int data_im_offset = d_iter[0] * shared_dilation[0];
        for (i = 1; i < num_axes; ++i) {
          data_im_offset *= shared_im_shape[i + 1];
          data_im_offset += d_iter[i] * shared_dilation[i];
        }
        *data_col_ptr = data_im_ptr[data_im_offset];
      } else {
        *data_col_ptr = 0;
      }
      data_col_ptr += data_col_inc;
      incremented = false;
      for (i = num_axes - 1; i >= 0; --i) {
        const int d_max = shared_kernel_shape[i];
        if (d_iter[i] == d_max - 1) {
          d_iter[i] = 0;
        } else {  // d_iter[i] < d_max - 1
          ++d_iter[i];
          incremented = true;
          break;
        }
      }  // for (int i = num_axes - 1; i >= 0; --i)
    } while (incremented);  // do
  }  // CUDA_KERNEL_LOOP(index, n)
}

template <typename Dtype>
void im2col_nd_gpu(const Dtype* data_im, const int num_spatial_axes,
    const int num_kernels, const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_col) {
  // num_axes should be smaller than block size
  DCHECK_LT(num_spatial_axes, CAFFE_CUDA_NUM_THREADS);
  switch (num_spatial_axes) {
  case 1:
    im2col_nd_gpu_kernel<Dtype, 1>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  case 2:
    im2col_nd_gpu_kernel<Dtype, 2>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  case 3:
    im2col_nd_gpu_kernel<Dtype, 3>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  case 4:
    im2col_nd_gpu_kernel<Dtype, 4>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  case 5:
    im2col_nd_gpu_kernel<Dtype, 5>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  case 6:
    im2col_nd_gpu_kernel<Dtype, 6>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  case 7:
    im2col_nd_gpu_kernel<Dtype, 7>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  case 8:
    im2col_nd_gpu_kernel<Dtype, 8>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  case 9:
    im2col_nd_gpu_kernel<Dtype, 9>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  case 10:
    im2col_nd_gpu_kernel<Dtype, 10>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  default:
    LOG(FATAL) << "im2col_nd_gpu does not support computation with "
               << num_spatial_axes << " spatial axes";
  }
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void im2col_nd_gpu<float>(const float* data_im,
    const int num_spatial_axes, const int col_size,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, float* data_col);
template void im2col_nd_gpu<double>(const double* data_im,
    const int num_spatial_axes, const int col_size,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, double* data_col);

template <typename Dtype>
__global__ void col2im_gpu_kernel(const int n, const Dtype* data_col,
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    const int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = min(w_im / stride_w + 1, width_col);
    const int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = min(h_im / stride_h + 1, height_col);
    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = (h_im - h_col * stride_h);
        int w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                                height_col + h_col) * width_col + w_col;
          val += data_col[data_col_index];
        }
      }
    }
    data_im[index] = val;
  }
}

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    Dtype* data_im) {
  int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) /
      stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) /
      stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, channels, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      height_col, width_col, data_im);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void col2im_gpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_im);
template void col2im_gpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    double* data_im);

template <typename Dtype, int num_axes>
__global__ void col2im_nd_gpu_kernel(const int n, const Dtype* data_col,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_im) {
  int d_im[num_axes];  // NOLINT(runtime/arrays)
  int d_col_iter[num_axes];  // NOLINT(runtime/arrays)
  int d_col_start[num_axes];  // NOLINT(runtime/arrays)
  int d_col_end[num_axes];  // NOLINT(runtime/arrays)

  __shared__ int shared_dilation[num_axes];
  __shared__ int shared_kernel_shape[num_axes];
  __shared__ int shared_pad[num_axes];
  __shared__ int shared_stride[num_axes];
  __shared__ int shared_col_shape[num_axes + 1];
  __shared__ int shared_im_shape[num_axes + 1];

  if (threadIdx.x < num_axes) {
    shared_dilation[threadIdx.x] = dilation[threadIdx.x];
    shared_kernel_shape[threadIdx.x] = kernel_shape[threadIdx.x];
    shared_pad[threadIdx.x] = pad[threadIdx.x];
    shared_stride[threadIdx.x] = stride[threadIdx.x];
  }
  if (threadIdx.x < num_axes + 1) {
    shared_col_shape[threadIdx.x] = col_shape[threadIdx.x];
    shared_im_shape[threadIdx.x] = im_shape[threadIdx.x];
  }
  __syncthreads();

  CUDA_KERNEL_LOOP(index, n) {
    // Initialize channel_in, computed in the loop below, with intermediate
    // computations used to compute the spatial indices.
    int c_im = index;
    // Calculate d_im (image dimensions).
    for (int i = num_axes - 1; i >= 0; --i) {
      d_im[i] = c_im % shared_im_shape[i + 1] + shared_pad[i];
      c_im /= shared_im_shape[i + 1];
    }
    // Calculate col start/end indices.
    bool done = false;
    for (int i = 0; i < num_axes; ++i) {
      const int kernel_extent =
          shared_dilation[i] * (shared_kernel_shape[i] - 1) + 1;
      d_col_start[i] = d_col_iter[i] =
          (d_im[i] < kernel_extent) ? 0 :
          (d_im[i] - kernel_extent) / shared_stride[i] + 1;
      d_col_end[i] =
          min(d_im[i] / shared_stride[i] + 1, shared_col_shape[i + 1]);
      if (d_col_start[i] >= d_col_end[i]) {
        // Skip computation if the dimension is 0 at any spatial axis --
        // final val will be 0.
        data_im[index] = 0;
        done = true;
        break;  // for (int i = 0; i < num_axes; ++i)
      }
    }
    if (done) {
      continue;  // CUDA_KERNEL_LOOP(index, n)
    }
    // Loop over the col to compute the output val.
    Dtype val = 0;
    bool incremented = true;
    bool skip = false;
    do {
      // Compute the final offset.
      int final_offset = 0;
      int kernel_shape_prod = 1;
      int kernel_index;
      for (int i = num_axes - 1; i >= 0; --i) {
        kernel_index = d_im[i] - d_col_iter[i] * shared_stride[i];
        if (kernel_index % shared_dilation[i]) {
          skip = true;
          break;
        } else {
          kernel_index /= shared_dilation[i];
          final_offset += kernel_index * kernel_shape_prod;
          kernel_shape_prod *= shared_kernel_shape[i];
        }
      }
      if (!skip) {
        final_offset += kernel_shape_prod * c_im;
        for (int i = 0; i < num_axes; ++i) {
          final_offset *= shared_col_shape[i + 1];
          final_offset += d_col_iter[i];
        }
        val += data_col[final_offset];
      }
      skip = false;
      incremented = false;
      for (int i = num_axes - 1; i >= 0; --i) {
        const int d_max = d_col_end[i];
        if (d_col_iter[i] == d_max - 1) {
          d_col_iter[i] = d_col_start[i];
        } else {  // d_col_iter[i] < d_max - 1
          ++d_col_iter[i];
          incremented = true;
          break;  // for (int i = num_axes - 1; i >= 0; --i)
        }
      }  // for (int i = num_axes - 1; i >= 0; --i)
    }  while (incremented);
    data_im[index] = val;
  }  // CUDA_KERNEL_LOOP(index, n)
}

template <typename Dtype>
void col2im_nd_gpu(const Dtype* data_col, const int num_spatial_axes,
    const int im_size, const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_im) {
  // num_axes should be smaller than block size
  DCHECK_LT(num_spatial_axes, CAFFE_CUDA_NUM_THREADS);
  switch (num_spatial_axes) {
  case 1:
    col2im_nd_gpu_kernel<Dtype, 1>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  case 2:
    col2im_nd_gpu_kernel<Dtype, 2>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  case 3:
    col2im_nd_gpu_kernel<Dtype, 3>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  case 4:
    col2im_nd_gpu_kernel<Dtype, 4>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  case 5:
    col2im_nd_gpu_kernel<Dtype, 5>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  case 6:
    col2im_nd_gpu_kernel<Dtype, 6>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  case 7:
    col2im_nd_gpu_kernel<Dtype, 7>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  case 8:
    col2im_nd_gpu_kernel<Dtype, 8>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  case 9:
    col2im_nd_gpu_kernel<Dtype, 9>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  case 10:
    col2im_nd_gpu_kernel<Dtype, 10>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  default:
    LOG(FATAL) << "col2im_nd_gpu does not support computation with "
               << num_spatial_axes << " spatial axes";
  }
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void col2im_nd_gpu<float>(const float* data_col,
    const int num_spatial_axes, const int im_size,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, float* data_im);
template void col2im_nd_gpu<double>(const double* data_col,
    const int num_spatial_axes, const int im_size,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, double* data_im);

}  // namespace caffe
