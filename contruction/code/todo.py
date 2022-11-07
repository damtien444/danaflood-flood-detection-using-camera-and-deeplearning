# done: cài đặt mô hình base và pipeline cơ bản
#       thu thập dữ liệu và đánh nhãn dữ liệu segmentation lũ (sử dụng công cụ iterative segementation training)
#       tìm hiểu về các loss trong bài toán segmentation,
#           theo bài báo Zaffaroni, M. and Rossi, C. (2022). Water Segmentation with Deep Learning Models for Flood Detection and Monitoring
#           thì sử dụng Tversky loss function (Salehi et al. 2017) with U = 0.2 and V = 0.8
#       xuất báo cáo thử nghiệm với model Unet gốc và BCEWithLogitsLoss
#       cài đặt loss function https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook

# done:
#       gán label độ sâu cho data -> với độ mịn 5 level (deadline: 27/10) -> done
#       cài đặt model chuẩn trong paper của thụy sỹ -> đã xong
#       tạo tập test từ dữ liệu camera (size 300) -> đã có
#       viết function test -> done
#       thử nghiệm Tversky Loss -> done -> not use
#       track loss value của train/dev/test để kiểm tra vấn đề -> done
#       training model với classify output -> lỗi cần sửa chữa -> done
#       điều chỉnh output classifier -> thêm convolution layer, thêm connected layer -> done

# todo:
#       chuẩn hóa input cho model? (chỉ resize) -> thêm bước tiền xử lý đã propose
#           https://learnopencv.com/otsu-thresholding-with-opencv/ -.> otsu threshold
#       fps tốc độ thực thi của mô hình? -> done
#       parse live video, record fps figure -> done
#       thêm data đường bình thường -> tuning model -> rechosen data
#       retrain lại thử nghiệm
#       thử nghiệm với combine loss
#       bắt đầu viết song song với
