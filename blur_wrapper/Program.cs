﻿using System;
using System.IO;
using Darknet;
using static Darknet.YoloWrapper;
using static Darknet.IdWrapper;
using ResNet;
using static ResNet.ResNet18Wrapper;
using Common;

using OpenCvSharp;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Drawing;
using System.Linq;
using System.Globalization;
using System.Text;

namespace YoloTest_CS
{
    class YoloResult
    {
        public int x { get; set; }
        public int y { get; set; }
        public int w { get; set; }
        public int h { get; set; }
        public float prob { get; set; }
        public int obj_id { get; set; }
    }

    class Program
    {
        private static readonly NLog.Logger logger = NLog.LogManager.GetCurrentClassLogger();
        static Dictionary<int, string> _namesDict = new Dictionary<int, string>
        {
            { 0, "1" }, { 1, "2" }, { 2, "3" }, { 3, "4" }, { 4, "5" },
            { 5, "6" }, { 6, "7" }, { 7, "8" }, { 8, "9" }, { 9, "0" },
            { 10, "A" }, { 11, "E" }, { 12, "F" }, { 13, "H" }, { 14, "J" },
            { 15, "K" }, { 16, "P" }, { 17, "X" }, { 18, "Y" }, { 19, "L" },
            { 20, "B" }, { 21, "T" }, { 22, "R" }
        };
        static YoloWrapper roiDetector;
        static IdWrapper idDetector;
        static ResNet18Wrapper resnetClassificator;
        const float conf_thres = 0.25f;
        static string summary_csv = @"\\172.16.0.24\工艺工程部设备运维\图片识别\Blur\res\summary.csv";
        static float res_conf_thres = float.Parse(DataPathcs.GetConnectionStringsConfig("BlurConfThresh"));

        public static byte[] GetPictureData(string imagePath)
        {
            FileStream fs = new FileStream(imagePath, FileMode.Open);
            byte[] byteData = new byte[fs.Length];
            fs.Read(byteData, 0, byteData.Length);
            fs.Close();
            return byteData;
        }

        static void Main(string[] args)
        {
            string path = args[0];
            string output_path = args[1];
            if (args.Length > 2)
                summary_csv = args[2];

            DirectoryInfo root = new DirectoryInfo(path);
            FileInfo[] file_info_list = root.GetFiles("*.jpg", SearchOption.AllDirectories);

            //load roi detection model yolov5s, and blur classification model resnet18
            DateTime begin_load_model = DateTime.Now;
            try
            {
                // keep model config name as xxx_yolov5s
                roiDetector = new YoloWrapper("roi_yolov5s", "../models/roi_yolov5s.wts", 0, 5);
                idDetector = new IdWrapper("id_yolov5s", "../models/id_yolov5s.wts", 0, 23);
                resnetClassificator = new ResNet18Wrapper("resnet18", "../models/resnet18.wts", 0);
            }
            catch (Exception)
            {
                throw;
            }
            DateTime end_load_model = DateTime.Now;
            logger.Info("+++++++++++++++++++++++++++++++++++");
            logger.Info($"Time consumed for loading model: {(end_load_model - begin_load_model).TotalMilliseconds} ms");
            logger.Info("+++++++++++++++++++++++++++++++++++");

            DateTime begin_detect_class = DateTime.Now;
            if (Directory.Exists(output_path))
            {
                Directory.Delete(output_path, true);
            }
            Directory.CreateDirectory(output_path);
            Directory.CreateDirectory($"{output_path}//empty");
            Directory.CreateDirectory($"{output_path}//notBlurry");
            Directory.CreateDirectory($"{output_path}//blurry");
            if (!File.Exists(summary_csv))
                using (StreamWriter writer = new StreamWriter(summary_csv))
                    writer.WriteLine("ID,Line,Date,Status");
            foreach( var file_info in file_info_list)
            {
                if (file_info.Name.Contains("12345"))
                    continue;
                DetectROIClassBlur(file_info.DirectoryName, file_info.Name, output_path, true);
                logger.Info("****************************************");
                logger.Info("");
            }
            DateTime end_detect_class = DateTime.Now;
            logger.Info("+++++++++++++++++++++++++++++++++++");
            logger.Info($"Time consumed for evaluate {file_info_list.Length} images: {(end_detect_class - begin_detect_class).TotalMilliseconds} ms");
            logger.Info("+++++++++++++++++++++++++++++++++++");


            //dispose of the two models
            roiDetector.Dispose();
            idDetector.Dispose();
            resnetClassificator.Dispose();
        } 


        public static void DetectROIClassBlur(string path, string image_name, string output_path, bool save_plot=false)
        {
            logger.Info($"image file name: {path}//{image_name}");

            //read images
            Mat img = Cv2.ImRead($"{path}//{image_name}");
            Cv2.Transpose(img, img);
            Cv2.Flip(img, img, FlipMode.Y);
            
            byte[] picBytes = img.ToBytes();

            //detection of roi using yolov5s
            bbox_t[] result_list;
            DateTime begin_roi = DateTime.Now;
            lock(roiDetector)
            {
                result_list = roiDetector.Detect(picBytes);
            }
            DateTime end_roi = DateTime.Now;
            logger.Info($"Time consumed for detect roi: {(end_roi - begin_roi).TotalMilliseconds} ms");

            List<YoloResult> roiResults = new List<YoloResult>();

            //convert roi result from bbox_t[] to YoloResult
            foreach (var b in result_list)
            {
                if (b.h == 0)
                {
                    break;
                }
                if (b.prob < conf_thres)
                {
                    continue;
                }
                roiResults.Add(new YoloResult
                {
                    x = (int)b.x,
                    y = (int)b.y,
                    w = (int)b.w,
                    h = (int)b.h,
                    prob = b.prob,
                    obj_id = (int)b.obj_id
                });
            }

            Mat plot_img = null;
            string image_id = image_name;
            if (save_plot)
            {
                plot_img = img.Clone();
            }
            //start blur classification
            int imageStatus = 0;
            string statusPath = "notBlurry";
            logger.Info($"start blurry classification with length {roiResults.Count}");
            if (roiResults.Count == 0)
            {
                logger.Info("Empty plate");
                imageStatus = -1;
                statusPath = "empty";
                Cv2.ImWrite($"{output_path}//{statusPath}//{image_id}", img);
                logger.Info($"{image_id} is checked to be {imageStatus}");
                return;
            }
            foreach (var result in roiResults)
            {
                if (result.obj_id == 0)
                {
                    logger.Info($"roi deteteted -> x:{result.x}, y:{result.y}, w:{result.w}, h:{result.h}");

                    //crop the two patches using roi result from yolov5s in turn
                    var roi = new OpenCvSharp.Rect(result.x, result.y, result.w, result.h);
                    Mat patch = new Mat(img, roi);

                    byte[] patchBytes = patch.ToBytes();
                    float conf;
                    DateTime begin_blur = DateTime.Now;
                    lock(resnetClassificator){
                        conf = resnetClassificator.ClassBlur(patchBytes);
                    }
                    DateTime end_blur = DateTime.Now;
                    logger.Info($"Time consumed for classify blur: {(end_blur - begin_blur).TotalMilliseconds} ms");
                    bool isBlurry = false;
                    if ( conf > res_conf_thres)
                    {
                        isBlurry = true;
                        imageStatus = 1;
                        statusPath = "blurry";
                    }
                    logger.Info($"blurry -> {isBlurry} with confidence -> {conf}");
                    save_plot = save_plot || isBlurry;
                    //plot of roi detection
                    if (save_plot)
                    {
                        if(plot_img == null)
                        {
                            plot_img = img.Clone();
                        }
                        plot_img.Rectangle(new OpenCvSharp.Rect(result.x, result.y, result.w, result.h), Scalar.Red, 3);
                        plot_img.PutText(isBlurry.ToString() + " " + conf.ToString("0.00"), new OpenCvSharp.Point(result.x, result.y+100), HersheyFonts.Italic, 4, Scalar.Red, 4);
                    }
                }
                else
                {
                    logger.Info($"id plate deteteted -> x:{result.x}, y:{result.y}");

                    //crop the id plate patches using roi result from yolov5s
                    var roi = new OpenCvSharp.Rect(result.x, result.y, result.w, result.h);
                    Mat patch = new Mat(img, roi);
                    Cv2.Transpose(patch, patch);
                    Cv2.Flip(patch, patch, 0);
                    byte[] patchBytes = patch.ToBytes();

                    //detection of id using yolov5s
                    bbox_t[] id_result_list;
                    DateTime begin_id = DateTime.Now;
                    lock(idDetector)
                    {
                        id_result_list = idDetector.Detect(patchBytes);
                    }
                    DateTime end_id = DateTime.Now;
                    logger.Info($"Time consumed for detect id: {(end_id - begin_id).TotalMilliseconds} ms");


                    image_id = ConvertData(id_result_list);
                    logger.Info(image_id);
                    if (image_id.Length != 8)
                    {
                        Cv2.ImWrite($"..//id_patch//{image_id}.jpg", patch);
                    }
                }
            }
            logger.Info($"{image_id} is checked to be {imageStatus}");

            if (save_plot)
            {
                if(statusPath=="blurry")
                    Cv2.ImWrite($"{output_path}/{statusPath}/{image_id}.jpg", plot_img);


                int start_idx = image_name.LastIndexOf('-')+1;
                int end_idx = image_name.LastIndexOf('.');
                string pressLine = new DirectoryInfo(path).Name;
                string date_s = image_name.Substring(start_idx, end_idx - start_idx);

                var date = DateTime.ParseExact(date_s, "yyyyMMddHHmmss", CultureInfo.InvariantCulture);
                string line = $"{image_id},{pressLine},{date},{statusPath}";
                using (StreamWriter writer = new StreamWriter(summary_csv,true))
                    writer.WriteLine(line);
            }
        }

        public static string ConvertData(bbox_t[] bbox)
        {
            List<YoloResult> boundingBoxes = new List<YoloResult>();
            List<uint> y = new List<uint>();
            var id = "";
            //var table = new ConsoleTable("Type", "Confidence", "X", "Y", "Width", "Height");
            foreach (var item in bbox.Where(o => o.h > 0 || o.w > 0))
            {
                //var type = _namesDict[(int)item.obj_id];
                //table.AddRow(type, item.prob, item.x, item.y, item.w, item.h);
                boundingBoxes.Add(new YoloResult
                {
                    x = (int)item.x,
                    y = (int)item.y,
                    w = (int)item.w,
                    h = (int)item.h,
                    prob = item.prob,
                    obj_id = (int)item.obj_id
                });
                y.Add(item.y);
                logger.Info(item.y);
            }
            //table.Write(Format.MarkDown);
            if (boundingBoxes.Count == 0)
            {
                return "NoRead";
            }
            var y_3ave = ((uint)y.Average(m => Convert.ToInt32(m))) / 4;
            var y_ave = (uint)y.Average(m => Convert.ToInt32(m));
            var line_one = boundingBoxes.FindAll(a => a.y < y_ave);
            line_one.Sort((m, n) => m.x.CompareTo(n.x));
            var line_two = boundingBoxes.FindAll(a => a.y > y_ave);
            line_two.Sort((m, n) => m.x.CompareTo(n.x));
            foreach (var item in line_one)
            {
                id += _namesDict[item.obj_id];
            }
            foreach (var item in line_two)
            {
                id += _namesDict[item.obj_id];
            }
            logger.Info($"Image ID is: {id}");
            return id;
        }
    }

}
