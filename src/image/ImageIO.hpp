#ifndef IMAGEIO_HPP
#define IMAGEIO_HPP

#include <map>
#include <string>
#include <vector>


namespace cv {
class Mat;
}

/**
 * @brief Loads a single image from a file.
 * @param filename The path to the image file.
 * @param flags The flags for loading the image (default is 1).
 * @return The loaded image as a cv::Mat object.
 */
auto loadImage(const std::string &filename, int flags = 1) -> cv::Mat;

/**
 * @brief Loads all images from a folder.
 * @param folder The path to the folder containing the images.
 * @param filenames The list of filenames to load (optional).
 * @param flags The flags for loading the images (default is 1).
 * @return A vector of pairs containing the filename and the loaded image as a
 * cv::Mat object.
 */
auto loadImages(const std::string &folder,
                const std::vector<std::string> &filenames = {}, int flags = 1)
    -> std::vector<std::pair<std::string, cv::Mat>>;

/**
 * @brief Saves an image to a file.
 * @param filename The path to the output image file.
 * @param image The image to save.
 * @return True if the image was saved successfully, false otherwise.
 */
auto saveImage(const std::string &filename, const cv::Mat &image) -> bool;

/**
 * @brief Saves a cv::Mat image as an 8-bit JPG file.
 * @param image The image to save.
 * @param output_path The path to the output JPG file (default is
 * "/dev/shm/MatTo8BitJPG.jpg").
 * @return True if the image was saved successfully, false otherwise.
 */
auto saveMatTo8BitJpg(const cv::Mat &image, const std::string &output_path =
                                                "/dev/shm/MatTo8BitJPG.jpg")
    -> bool;

/**
 * @brief Saves a cv::Mat image as a 16-bit PNG file.
 * @param image The image to save.
 * @param output_path The path to the output PNG file (default is
 * "/dev/shm/MatTo16BitPNG.png").
 * @return True if the image was saved successfully, false otherwise.
 */
auto saveMatTo16BitPng(const cv::Mat &image, const std::string &output_path =
                                                 "/dev/shm/MatTo16BitPNG.png")
    -> bool;

/**
 * @brief Saves a cv::Mat image as a FITS file.
 * @param image The image to save.
 * @param output_path The path to the output FITS file (default is
 * "/dev/shm/MatToFITS.fits").
 * @return True if the image was saved successfully, false otherwise.
 */
auto saveMatToFits(const cv::Mat &image,
                   const std::string &output_path = "/dev/shm/MatToFITS.fits")
    -> bool;

/**
 * @brief Retrieves metadata from a FITS image file.
 * @param filepath The path to the FITS file.
 * @return A map containing the metadata key-value pairs.
 */
auto getFitsMetadata(const std::string &filepath)
    -> std::map<std::string, std::string>;

#endif // IMAGEIO_HPP