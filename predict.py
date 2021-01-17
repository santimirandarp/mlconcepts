
def predict(w, b, X_test, Y_test, classes):
   pred = sigmoid(np.dot(w, X_test)+b)
   total = len(pred[0])
   correct = 0
   for i in range(total):
       prediction = classes(pred[0][i])
       real = classes(Y[0][i])
       print("you predicted", prediction )
       print("it is a", real )
       if(prediction==real):
          correct+=1
  
