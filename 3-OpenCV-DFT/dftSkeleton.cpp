#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

///helping functions
Mat rotateImage(Mat src, float angle) {
    // Computing the center of the image
    Point2f src_center(src.cols/2.0f, src.rows/2.0f);
    // We rotate around the center from the angle 'angle' (in degrees)
    Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
    Mat dst;
    // We apply the affiner transformation
    warpAffine(src, dst, rot_mat, src.size());
    
    // Returning result
    return dst;
}


Mat rotateResizeImage(Mat src, float angle) {
    // Computing the center of the image
    Point2f src_center(src.cols/2.0f, src.rows/2.0f);
    // We rotate around the center from the angle 'angle' (in degrees)
    Mat rot = getRotationMatrix2D(src_center, angle, 1.0);
    
    // We compute the bounding rectangle of the rotated image (in order not to 'crop' the image)
    cv::Rect bbox = cv::RotatedRect(src_center,src.size(), angle).boundingRect();
    // adjust transformation matrix
    rot.at<double>(0,2) += bbox.width/2.0 - src_center.x;
    rot.at<double>(1,2) += bbox.height/2.0 - src_center.y;
    
    // Apply the transformation
    Mat dst;
    warpAffine(src, dst, rot, bbox.size());
    
    // Returning result
    return dst;
}


/// Original OpenCV example code (refactored)
static void help()
{
    printf("\nThis program demonstrated the use of the discrete Fourier transform (dft)\n"
           "The dft of an image is taken and it's power spectrum is displayed.\n"
           "Usage:\n"
           "./dft [image_name -- default ../imagesDeTest/Baboon.jpg]\n");
}

const char* keys =
{
    "{@image|../imagesDeTest/Baboon.jpg|input image file}"
};

/// Computing the DFT within a function: we keep only the magnitude part of the DFT
/// detailled explanation of this code is here : https://docs.opencv.org/4.5.5/d8/d01/tutorial_discrete_fourier_transform.html
Mat dft(Mat img)
{
    // Computing the optimal size in case the image does not have a 'correct size'
    Mat padded; //expand input image to optimal size
    int M = getOptimalDFTSize( img.rows );
    int N = getOptimalDFTSize( img.cols ); // on the border add zero values
    copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));
    
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexImg;
    merge(planes, 2, complexImg);  // Add to the expanded another plane with zeros
    
    dft(complexImg, complexImg); // this way the result may fit in the source matrix
    
    // compute the magnitude and switch to logarithmic scale
    // compute log(1 + sqrt(Re(DFT(img))**2 + Im(DFT(img))**2))
    split(complexImg, planes);                  // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]); // planes[0] = magnitude
    Mat magI = planes[0].clone();
    
    magI += Scalar::all(1); // switch to logarithmic scale
    log(magI, magI);
    
    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;
  
    Mat q0(magI, Rect(0, 0, cx, cy)); // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy)); // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy)); // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
    
    Mat tmp;        // swap quadrants (Top-Left with Bottom-Right)

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    
    q1.copyTo(tmp); // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
    
    normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).
    return magI;
}


// Computing the DFT within a function
Mat dftComplex(Mat img)
{
    // Computing the optimal size in case the image does not have a 'correct size'
    int M = getOptimalDFTSize( img.rows );
    int N = getOptimalDFTSize( img.cols );
    Mat padded;
    copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));
    
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexImg;
    merge(planes, 2, complexImg);
    
    dft(complexImg, complexImg,cv::DFT_SCALE|cv::DFT_COMPLEX_OUTPUT);
    
    return complexImg;
}


// Inverse DFT
// We suppose the input corresponds to a correct (!) DFT
Mat idft(Mat fourierTransform) {
    
    // IFDT
    cv::Mat inverseTransform;
    cv::dft(fourierTransform, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    
    // Back to 8-bits
    cv::Mat finalImage;
    inverseTransform.convertTo(inverseTransform, CV_8U);
    
    return inverseTransform;
}


// Main function
int main(int argc, const char ** argv)
{
    CommandLineParser parser(argc, argv, keys);
    string filename = parser.get<string>(0);
    
    Mat img = imread(filename.c_str(), IMREAD_GRAYSCALE);
    if( img.empty() )
    {
        help();
        printf("Cannot read image file: %s\n", filename.c_str());
        return -1;
    }
    
    imshow("Original image", img);
    
    Mat mag = dft(img);
    imshow("Spectrum magnitude", mag);
    
    // A vous de faire les appels necessaire pour
    // 1) appliquer une rotation de 45 degres à l'image
    // 2) appliquer la meme rotation en redimensionnant l'image
    // Indice : regardez les fonctions ecrites plus haut
    ACOMPLETER
    
    
    // Montrez les deux images resultat
    imshow("Rotated Img", ACOMPLETER);
    imshow("Rotated resize", ACOMPLETER);
    
    // Calcul de la DFT (en amplitude seulement) des images tournees
    Mat magRot = dft(ACOMPLETER);
    Mat magRotResize = dft(ACOMPLETER);
    
    // Montrez le resultat
    imshow("Spectrum Rotated", magRot);
    imshow("Spectrum Rotated Resized", magRotResize);
    
    // On calcule ici la DFT inverse de l'image originale
    Mat invImg;
    Mat testComplex = dftComplex(img);
    invImg = idft(testComplex);
    // et on montre le resultat
    imshow("Inverse DFT", invImg);
    
    // Faites la meme chose pour les images tournees
    ACOMPLETER
    
    
    waitKey();
    return 0;
}
