# ANN_ParallelComputing

## 概述

**ANN_ParallelComputing** 是一個深度學習項目，旨在探索簡單的類神經網絡 (ANN) 架構，並通過 OpenMP 和 OpenCL 實現平行運算，以提高訓練和推斷的性能。這個項目專注於使用 MNIST 資料集和 CIFAR-10 資料集來進行深度學習應用。

## 功能和目標

- 使用簡單的 ANN 架構來進行深度學習。
- 利用 OpenMP 實現多核 CPU 上的平行化，以加速訓練過程。
- 利用 OpenCL 實現 GPU 或其他加速器上的平行化，以進一步提高性能。
- 提供 MNIST 和 CIFAR-10 的訓練和測試功能。
- 支援自定義的網絡配置和超參數調整。

## 主要功能

`mnist.hpp` 檔案包含了與 MNIST 資料集相關的功能和結構定義。MNIST 資料集是一個經典的手寫數字識別資料集，通常用於深度學習和機器學習的測試和實驗。  
`readCIFAR10.hpp` 檔案包含了 `readCIFAR10` 函數的定義，這是一個用於讀取 CIFAR-10 資料集的功能。CIFAR-10 資料集是一個常用的圖像分類測試資料集，其中包含多個類別的圖像。  
`initGPU.hpp` 檔案包含了用於初始化 GPU 資源的功能，並使用 OpenCL 作為 GPU 運算的庫。這個程式庫有助於配置和準備 GPU 以進行平行計算。  
`MNIST.hpp` 檔案包含了用於處理 MNIST 資料集圖像數據的功能和結構定義。MNIST 資料集是一個經典的手寫數字識別測試資料集，通常用於深度學習和機器學習實驗。  

`ANN.cpp` 檔案包含了一個神經網絡（Artificial Neural Network，ANN）訓練程式的實現。這個程式用於構建、訓練和評估 ANN 模型，並使用反向傳播算法進行訓練。  
`ANNOCL.cpp` 檔案為ANN.cpp導入OpenCL之實作。  
`ANNOMP.cpp` 檔案為ANN.cpp導入OpenMP之實作。  
`CIFAR10.cpp` 檔案包含了一個用於讀取CIFAR-10數據集的C++函數。  
`GPU.cpp` 檔案包含了使用OpenCL執行GPU計算的功能。  
`main.cpp` 檔案包含了用於實現深度學習自動編碼器（Autoencoder）的範例。  
`MNIST.cpp` 檔案包含了一個用於讀取MNIST數據集的C++函數。  

## 致謝

此專案受到 謝佑明教授 的啟發和支持，感謝他提供的檔案和教學。本專案僅用於學習和教育目的，未用於任何商業用途。  
