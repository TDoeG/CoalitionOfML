# Expermiment 1
"""Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./assets/cifar-10\cifar-10-python.tar.gz
100%|██████████| 170498071/170498071 [00:33<00:00, 5124972.32it/s]
Extracting ./assets/cifar-10\cifar-10-python.tar.gz to ./assets/cifar-10
Training Progress:   0%|          | 0/100 [00:00<?, ?epoch/s]c:\Users\tyler\anaconda3\envs\grayscale_env\lib\site-packages\torch\nn\modules\loss.py:535: UserWarning: Using a target size (torch.Size([3, 32, 32])) that is different to the input size (torch.Size([1, 3, 32, 32])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
  return F.mse_loss(input, target, reduction=self.reduction)
Training Progress:   1%|          | 1/100 [07:56<13:06:13, 476.50s/epoch]
EPOCH: 1 | Train_loss: 8.0146 | Test_loss: 0.0046
Training Progress:   2%|▏         | 2/100 [15:37<12:43:10, 467.25s/epoch]
EPOCH: 2 | Train_loss: 0.0159 | Test_loss: 0.0031
Training Progress:   3%|▎         | 3/100 [23:19<12:31:54, 465.10s/epoch]
EPOCH: 3 | Train_loss: 0.0123 | Test_loss: 0.0028
Training Progress:   4%|▍         | 4/100 [30:59<12:20:36, 462.88s/epoch]
EPOCH: 4 | Train_loss: 0.0110 | Test_loss: 0.0026
Training Progress:   5%|▌         | 5/100 [38:40<12:11:40, 462.11s/epoch]
EPOCH: 5 | Train_loss: 0.0102 | Test_loss: 0.0024
Training Progress:   6%|▌         | 6/100 [46:21<12:03:47, 462.00s/epoch]
EPOCH: 6 | Train_loss: 0.0097 | Test_loss: 0.0023
Training Progress:   7%|▋         | 7/100 [53:59<11:53:43, 460.47s/epoch]
EPOCH: 7 | Train_loss: 0.0093 | Test_loss: 0.0022
Training Progress:   8%|▊         | 8/100 [1:01:30<11:41:22, 457.42s/epoch]
EPOCH: 8 | Train_loss: 0.0090 | Test_loss: 0.0022
Training Progress:   9%|▉         | 9/100 [1:09:14<11:37:05, 459.62s/epoch]
EPOCH: 9 | Train_loss: 0.0087 | Test_loss: 0.0021
Training Progress:  10%|█         | 10/100 [1:16:49<11:27:06, 458.07s/epoch]
EPOCH: 10 | Train_loss: 0.0085 | Test_loss: 0.0021
Training Progress:  11%|█         | 11/100 [1:24:22<11:17:23, 456.67s/epoch]
EPOCH: 11 | Train_loss: 0.0083 | Test_loss: 0.0021
Training Progress:  12%|█▏        | 12/100 [1:32:02<11:11:01, 457.51s/epoch]
EPOCH: 12 | Train_loss: 0.0081 | Test_loss: 0.0020
Training Progress:  13%|█▎        | 13/100 [1:39:54<11:09:48, 461.94s/epoch]
EPOCH: 13 | Train_loss: 0.0080 | Test_loss: 0.0020
Training Progress:  14%|█▍        | 14/100 [1:47:32<11:00:22, 460.73s/epoch]
EPOCH: 14 | Train_loss: 0.0079 | Test_loss: 0.0020
Training Progress:  15%|█▌        | 15/100 [1:55:03<10:48:45, 457.95s/epoch]
EPOCH: 15 | Train_loss: 0.0078 | Test_loss: 0.0020
Training Progress:  16%|█▌        | 16/100 [2:02:36<10:39:07, 456.52s/epoch]
EPOCH: 16 | Train_loss: 0.0077 | Test_loss: 0.0019
Training Progress:  17%|█▋        | 17/100 [2:10:11<10:30:34, 455.83s/epoch]
EPOCH: 17 | Train_loss: 0.0076 | Test_loss: 0.0019
Training Progress:  18%|█▊        | 18/100 [2:17:43<10:21:26, 454.71s/epoch]
EPOCH: 18 | Train_loss: 0.0075 | Test_loss: 0.0019
Training Progress:  19%|█▉        | 19/100 [2:25:12<10:11:34, 453.02s/epoch]
EPOCH: 19 | Train_loss: 0.0074 | Test_loss: 0.0019
Training Progress:  20%|██        | 20/100 [2:32:39<10:01:44, 451.31s/epoch]
EPOCH: 20 | Train_loss: 0.0073 | Test_loss: 0.0018
Training Progress:  21%|██        | 21/100 [2:40:12<9:54:41, 451.67s/epoch] 
EPOCH: 21 | Train_loss: 0.0073 | Test_loss: 0.0018
Training Progress:  22%|██▏       | 22/100 [2:47:50<9:50:00, 453.86s/epoch]
EPOCH: 22 | Train_loss: 0.0072 | Test_loss: 0.0018
Training Progress:  23%|██▎       | 23/100 [2:55:17<9:39:40, 451.70s/epoch]
EPOCH: 23 | Train_loss: 0.0072 | Test_loss: 0.0018
Training Progress:  24%|██▍       | 24/100 [3:02:43<9:30:03, 450.04s/epoch]
EPOCH: 24 | Train_loss: 0.0071 | Test_loss: 0.0018
Training Progress:  25%|██▌       | 25/100 [3:10:09<9:20:46, 448.62s/epoch]
EPOCH: 25 | Train_loss: 0.0071 | Test_loss: 0.0018
Training Progress:  26%|██▌       | 26/100 [3:17:32<9:11:22, 447.06s/epoch]
EPOCH: 26 | Train_loss: 0.0070 | Test_loss: 0.0018
Training Progress:  27%|██▋       | 27/100 [3:24:55<9:02:31, 445.91s/epoch]
EPOCH: 27 | Train_loss: 0.0070 | Test_loss: 0.0017
Training Progress:  28%|██▊       | 28/100 [3:32:15<8:52:44, 443.95s/epoch]
EPOCH: 28 | Train_loss: 0.0069 | Test_loss: 0.0017
Training Progress:  29%|██▉       | 29/100 [3:39:35<8:44:08, 442.94s/epoch]
EPOCH: 29 | Train_loss: 0.0069 | Test_loss: 0.0017
Training Progress:  30%|███       | 30/100 [3:47:19<8:43:59, 449.13s/epoch]
EPOCH: 30 | Train_loss: 0.0068 | Test_loss: 0.0017
Training Progress:  31%|███       | 31/100 [3:54:55<8:38:50, 451.17s/epoch]
EPOCH: 31 | Train_loss: 0.0068 | Test_loss: 0.0017
Training Progress:  32%|███▏      | 32/100 [4:02:14<8:27:07, 447.47s/epoch]
EPOCH: 32 | Train_loss: 0.0068 | Test_loss: 0.0017
Training Progress:  33%|███▎      | 33/100 [4:09:33<8:16:51, 444.95s/epoch]
EPOCH: 33 | Train_loss: 0.0067 | Test_loss: 0.0017
Training Progress:  34%|███▍      | 34/100 [4:17:02<8:10:59, 446.36s/epoch]
EPOCH: 34 | Train_loss: 0.0067 | Test_loss: 0.0017
Training Progress:  35%|███▌      | 35/100 [4:24:29<8:03:45, 446.55s/epoch]
EPOCH: 35 | Train_loss: 0.0067 | Test_loss: 0.0017
Training Progress:  36%|███▌      | 36/100 [4:32:07<7:59:56, 449.94s/epoch]
EPOCH: 36 | Train_loss: 0.0067 | Test_loss: 0.0017
Training Progress:  37%|███▋      | 37/100 [4:39:28<7:49:27, 447.10s/epoch]
EPOCH: 37 | Train_loss: 0.0066 | Test_loss: 0.0017
Training Progress:  38%|███▊      | 38/100 [4:46:52<7:41:03, 446.19s/epoch]
EPOCH: 38 | Train_loss: 0.0066 | Test_loss: 0.0017
Training Progress:  39%|███▉      | 39/100 [4:54:31<7:37:46, 450.26s/epoch]
EPOCH: 39 | Train_loss: 0.0066 | Test_loss: 0.0017
Training Progress:  40%|████      | 40/100 [5:02:05<7:31:15, 451.25s/epoch]
EPOCH: 40 | Train_loss: 0.0066 | Test_loss: 0.0017
Training Progress:  41%|████      | 41/100 [5:09:34<7:23:12, 450.72s/epoch]
EPOCH: 41 | Train_loss: 0.0065 | Test_loss: 0.0017
Training Progress:  42%|████▏     | 42/100 [5:16:58<7:13:45, 448.71s/epoch]
EPOCH: 42 | Train_loss: 0.0065 | Test_loss: 0.0016
Training Progress:  43%|████▎     | 43/100 [5:24:23<7:04:59, 447.35s/epoch]
EPOCH: 43 | Train_loss: 0.0065 | Test_loss: 0.0016
Training Progress:  44%|████▍     | 44/100 [5:31:49<6:57:18, 447.11s/epoch]
EPOCH: 44 | Train_loss: 0.0065 | Test_loss: 0.0016
Training Progress:  45%|████▌     | 45/100 [5:39:11<6:48:26, 445.57s/epoch]
EPOCH: 45 | Train_loss: 0.0064 | Test_loss: 0.0016
Training Progress:  46%|████▌     | 46/100 [5:46:31<6:39:30, 443.89s/epoch]
EPOCH: 46 | Train_loss: 0.0064 | Test_loss: 0.0016
Training Progress:  47%|████▋     | 47/100 [5:53:57<6:32:39, 444.52s/epoch]
EPOCH: 47 | Train_loss: 0.0064 | Test_loss: 0.0016
Training Progress:  48%|████▊     | 48/100 [6:01:23<6:25:42, 445.05s/epoch]
EPOCH: 48 | Train_loss: 0.0064 | Test_loss: 0.0016
Training Progress:  49%|████▉     | 49/100 [6:08:52<6:19:10, 446.09s/epoch]
EPOCH: 49 | Train_loss: 0.0064 | Test_loss: 0.0016
Training Progress:  50%|█████     | 50/100 [6:16:20<6:12:10, 446.62s/epoch]
EPOCH: 50 | Train_loss: 0.0063 | Test_loss: 0.0016
Training Progress:  51%|█████     | 51/100 [6:23:47<6:04:53, 446.81s/epoch]
EPOCH: 51 | Train_loss: 0.0063 | Test_loss: 0.0016
Training Progress:  52%|█████▏    | 52/100 [6:31:09<5:56:15, 445.32s/epoch]
EPOCH: 52 | Train_loss: 0.0063 | Test_loss: 0.0016
Training Progress:  53%|█████▎    | 53/100 [6:38:36<5:49:15, 445.86s/epoch]
EPOCH: 53 | Train_loss: 0.0063 | Test_loss: 0.0016
Training Progress:  54%|█████▍    | 54/100 [6:46:01<5:41:31, 445.46s/epoch]
EPOCH: 54 | Train_loss: 0.0063 | Test_loss: 0.0016
Training Progress:  55%|█████▌    | 55/100 [6:53:26<5:34:01, 445.37s/epoch]
EPOCH: 55 | Train_loss: 0.0062 | Test_loss: 0.0016
Training Progress:  56%|█████▌    | 56/100 [7:00:45<5:25:18, 443.60s/epoch]
EPOCH: 56 | Train_loss: 0.0062 | Test_loss: 0.0016
Training Progress:  57%|█████▋    | 57/100 [7:08:03<5:16:40, 441.87s/epoch]
EPOCH: 57 | Train_loss: 0.0062 | Test_loss: 0.0016
Training Progress:  58%|█████▊    | 58/100 [7:15:26<5:09:34, 442.26s/epoch]
EPOCH: 58 | Train_loss: 0.0062 | Test_loss: 0.0016
Training Progress:  59%|█████▉    | 59/100 [7:22:49<5:02:17, 442.38s/epoch]
EPOCH: 59 | Train_loss: 0.0062 | Test_loss: 0.0016
Training Progress:  60%|██████    | 60/100 [7:30:12<4:54:59, 442.48s/epoch]
EPOCH: 60 | Train_loss: 0.0061 | Test_loss: 0.0016
Training Progress:  61%|██████    | 61/100 [7:37:44<4:49:31, 445.41s/epoch]
EPOCH: 61 | Train_loss: 0.0061 | Test_loss: 0.0016
Training Progress:  62%|██████▏   | 62/100 [7:45:11<4:42:21, 445.84s/epoch]
EPOCH: 62 | Train_loss: 0.0061 | Test_loss: 0.0016
Training Progress:  63%|██████▎   | 63/100 [7:52:31<4:33:54, 444.18s/epoch]
EPOCH: 63 | Train_loss: 0.0061 | Test_loss: 0.0016
Training Progress:  64%|██████▍   | 64/100 [7:59:55<4:26:31, 444.19s/epoch]
EPOCH: 64 | Train_loss: 0.0061 | Test_loss: 0.0016
Training Progress:  65%|██████▌   | 65/100 [8:07:18<4:18:50, 443.74s/epoch]
EPOCH: 65 | Train_loss: 0.0061 | Test_loss: 0.0016
Training Progress:  66%|██████▌   | 66/100 [8:14:40<4:11:12, 443.31s/epoch]
EPOCH: 66 | Train_loss: 0.0060 | Test_loss: 0.0016
Training Progress:  67%|██████▋   | 67/100 [8:22:10<4:04:56, 445.36s/epoch]
EPOCH: 67 | Train_loss: 0.0060 | Test_loss: 0.0016
Training Progress:  68%|██████▊   | 68/100 [8:29:37<3:57:46, 445.84s/epoch]
EPOCH: 68 | Train_loss: 0.0060 | Test_loss: 0.0016
Training Progress:  69%|██████▉   | 69/100 [8:37:04<3:50:30, 446.13s/epoch]
EPOCH: 69 | Train_loss: 0.0060 | Test_loss: 0.0016
Training Progress:  70%|███████   | 70/100 [8:45:27<3:51:33, 463.13s/epoch]
EPOCH: 70 | Train_loss: 0.0060 | Test_loss: 0.0015
Training Progress:  71%|███████   | 71/100 [8:53:47<3:49:09, 474.13s/epoch]
EPOCH: 71 | Train_loss: 0.0060 | Test_loss: 0.0015
Training Progress:  72%|███████▏  | 72/100 [9:02:58<3:52:04, 497.30s/epoch]
EPOCH: 72 | Train_loss: 0.0060 | Test_loss: 0.0016
Training Progress:  73%|███████▎  | 73/100 [9:09:37<3:30:32, 467.88s/epoch]
EPOCH: 73 | Train_loss: 0.0059 | Test_loss: 0.0016
Training Progress:  74%|███████▍  | 74/100 [9:16:01<3:11:47, 442.58s/epoch]
EPOCH: 74 | Train_loss: 0.0059 | Test_loss: 0.0016
Training Progress:  75%|███████▌  | 75/100 [9:22:21<2:56:33, 423.72s/epoch]
EPOCH: 75 | Train_loss: 0.0059 | Test_loss: 0.0016
Training Progress:  76%|███████▌  | 76/100 [9:28:38<2:43:57, 409.90s/epoch]
EPOCH: 76 | Train_loss: 0.0059 | Test_loss: 0.0015
Training Progress:  77%|███████▋  | 77/100 [9:34:56<2:33:25, 400.22s/epoch]
EPOCH: 77 | Train_loss: 0.0059 | Test_loss: 0.0016
Training Progress:  78%|███████▊  | 78/100 [9:41:13<2:24:10, 393.21s/epoch]
EPOCH: 78 | Train_loss: 0.0059 | Test_loss: 0.0016
Training Progress:  79%|███████▉  | 79/100 [9:47:29<2:15:50, 388.13s/epoch]
EPOCH: 79 | Train_loss: 0.0059 | Test_loss: 0.0016
Training Progress:  80%|████████  | 80/100 [9:53:46<2:08:18, 384.93s/epoch]
EPOCH: 80 | Train_loss: 0.0059 | Test_loss: 0.0016
Training Progress:  81%|████████  | 81/100 [10:00:04<2:01:09, 382.61s/epoch]
EPOCH: 81 | Train_loss: 0.0058 | Test_loss: 0.0015
Training Progress:  82%|████████▏ | 82/100 [10:06:22<1:54:26, 381.50s/epoch]
EPOCH: 82 | Train_loss: 0.0058 | Test_loss: 0.0015
Training Progress:  83%|████████▎ | 83/100 [10:12:42<1:47:53, 380.77s/epoch]
EPOCH: 83 | Train_loss: 0.0058 | Test_loss: 0.0015
Training Progress:  84%|████████▍ | 84/100 [10:19:00<1:41:19, 379.96s/epoch]
EPOCH: 84 | Train_loss: 0.0058 | Test_loss: 0.0015
Training Progress:  85%|████████▌ | 85/100 [10:25:20<1:34:59, 379.96s/epoch]
EPOCH: 85 | Train_loss: 0.0058 | Test_loss: 0.0015
Training Progress:  86%|████████▌ | 86/100 [10:31:38<1:28:31, 379.41s/epoch]
EPOCH: 86 | Train_loss: 0.0058 | Test_loss: 0.0015
Training Progress:  87%|████████▋ | 87/100 [10:37:57<1:22:13, 379.51s/epoch]
EPOCH: 87 | Train_loss: 0.0058 | Test_loss: 0.0015
Training Progress:  88%|████████▊ | 88/100 [10:44:21<1:16:10, 380.84s/epoch]
EPOCH: 88 | Train_loss: 0.0058 | Test_loss: 0.0015
Training Progress:  89%|████████▉ | 89/100 [10:50:39<1:09:39, 379.92s/epoch]
EPOCH: 89 | Train_loss: 0.0058 | Test_loss: 0.0015
Training Progress:  90%|█████████ | 90/100 [10:56:57<1:03:11, 379.16s/epoch]
EPOCH: 90 | Train_loss: 0.0058 | Test_loss: 0.0015
Training Progress:  91%|█████████ | 91/100 [11:03:15<56:51, 379.05s/epoch]  
EPOCH: 91 | Train_loss: 0.0057 | Test_loss: 0.0015
Training Progress:  92%|█████████▏| 92/100 [11:09:32<50:26, 378.25s/epoch]
EPOCH: 92 | Train_loss: 0.0057 | Test_loss: 0.0015
Training Progress:  93%|█████████▎| 93/100 [11:16:01<44:30, 381.55s/epoch]
EPOCH: 93 | Train_loss: 0.0057 | Test_loss: 0.0015
Training Progress:  94%|█████████▍| 94/100 [11:22:19<38:03, 380.59s/epoch]
EPOCH: 94 | Train_loss: 0.0057 | Test_loss: 0.0015
Training Progress:  95%|█████████▌| 95/100 [11:28:42<31:45, 381.06s/epoch]
EPOCH: 95 | Train_loss: 0.0057 | Test_loss: 0.0015
Training Progress:  96%|█████████▌| 96/100 [11:35:00<25:21, 380.42s/epoch]
EPOCH: 96 | Train_loss: 0.0057 | Test_loss: 0.0015
Training Progress:  97%|█████████▋| 97/100 [11:41:18<18:58, 379.60s/epoch]
EPOCH: 97 | Train_loss: 0.0057 | Test_loss: 0.0015
Training Progress:  98%|█████████▊| 98/100 [11:47:36<12:38, 379.14s/epoch]
EPOCH: 98 | Train_loss: 0.0057 | Test_loss: 0.0015
Training Progress:  99%|█████████▉| 99/100 [11:54:04<06:21, 381.64s/epoch]
EPOCH: 99 | Train_loss: 0.0057 | Test_loss: 0.0015
Training Progress: 100%|██████████| 100/100 [12:00:25<00:00, 432.25s/epoch]
EPOCH: 100 | Train_loss: 0.0057 | Test_loss: 0.0015
Finished Training
Model saved at ./saved_model\G2C_Exp0Epoch100.pth"""