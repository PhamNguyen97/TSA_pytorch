# TSA_pytorch
Training Signal Annealing

This is an example of TSA.
In the code above, I compare training a simple neural net of binary classification task:  
*Input:*  
```[a,b]```  
*Output:*  
```1 if a>b```  
```0 if a<b```  
  
## Training data  
```X_train = [11,10],[200,151],[501,400],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12],[12,13],[1,10],[9,20],[9,100],[100,101],[1111,1112]]```  
```Y_train = [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]```  

## Result  
```X_test = [[200,151],[501,400],[9,8],[99,95],[100,100000],[99,60]]```  
```Y_test = [1,1,1,1,0,1]```  
TSA_training:   
```output: [[0., 1.],[0., 1.],[0., 1.],[0., 1.],[1., 0.],[0., 1.]]```  
  
Without TSA_training:  
```output: [[0., 1.],[0., 1.],[0., 1.],[1., 0.],[1., 0.],[0., 1.]]```  
  

In the example, the number of "1"-class samples is much smaller than that of "0"-class samples.  
The simple test show that, the Net dose not recognize [99,95] input when it is trained in normal way while the same Net (with TSA training) can classify that input.  

## Conclution  
The code is an demo of applying TSA into crossEntopyLoss written in Pytorch.  
TSA make *loss surface* become *smoother* which means of *fewer local minimum*, helping the process of training neural net become easier, expecially in case of difficulty in data collection.  
The experiment implies that Neural Net can be trained with *unbalanced* dataset.   

## Reference  
Qizhe Xie, Zihang Dai, Eduard Hovy, Minh-Thang Luong, Quoc V. Le. Unsupervised Data Augmentation for Consistency Training https://arxiv.org/abs/1904.12848  
