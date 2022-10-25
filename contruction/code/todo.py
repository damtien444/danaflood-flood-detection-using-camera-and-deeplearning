# done: cài đặt mô hình base và pipeline cơ bản
#       thu thập dữ liệu và đánh nhãn dữ liệu segmentation lũ (sử dụng công cụ iterative segementation training)
#       tìm hiểu về các loss trong bài toán segmentation,
#           theo bài báo Zaffaroni, M. and Rossi, C. (2022). Water Segmentation with Deep Learning Models for Flood Detection and Monitoring
#           thì sử dụng Tversky loss function (Salehi et al. 2017) with U = 0.2 and V = 0.8



# todo: xuất báo cáo thử nghiệm với model Unet gốc và BCEWithLogitsLoss
#       gán label độ sâu cho data -> với độ mịn 5 level (deadline: 27/10)
#       cài đặt model chuẩn trong paper của thụy sỹ
#       cài đặt loss function https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook