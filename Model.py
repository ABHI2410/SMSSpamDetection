# Standared Imports
import os
import numpy as np
import matplotlib.pyplot as plt
#imports related to scikit-learn
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.cluster import KMeans
#imports related to tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
#Custom imports
from Data import Data

# Setting environment variable to ignor tensorflow warnings 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Model:
    def __init__(self) -> None:
        pass
    
    def svm_classifier(self,train,test):
        classifer = svm.SVC()
        classifer.fit(train[0],train[1])
        predictions = classifer.predict(test[0])

        cr = classification_report(test[1], predictions)
        cm = confusion_matrix(test[1], predictions)
        accuracy = classifer.score(test[0], test[1]) * 100
        fpr, tpr, _ = roc_curve(test[1], predictions)
        roc_auc = auc(fpr, tpr)
        return (cr,cm,accuracy, fpr, tpr, roc_auc)

    def kmeans_clustering(self,message_vector,label,seed):
        kmeans = KMeans(n_clusters=2, random_state=seed, n_init="auto").fit(message_vector)
        clusters = kmeans.labels_.tolist()

        cr = classification_report(label, clusters)
        cm = confusion_matrix(label, clusters)
        accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0]) * 100
        fpr, tpr, _ = roc_curve(label, clusters)
        roc_auc = auc(fpr, tpr)
        return (cr,cm,accuracy, fpr, tpr, roc_auc)
  

    def cnn(self,train,test):

        model = Sequential()
        model.add(Embedding(5000, 128))
        model.add(Conv1D(32, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

 
        model.fit(train[0], train[1], epochs=20, batch_size=32)
    

        loss, accuracy = model.evaluate(test[0], test[1])

        predictions = (model.predict(test[0]) > 0.5).astype("int32")

        cr = classification_report(test[1], predictions)

        cm = confusion_matrix(test[1], predictions)

        fpr, tpr, _ = roc_curve(test[1], predictions)
        roc_auc = auc(fpr, tpr)
        
        return (cr,cm,accuracy*100, fpr, tpr, roc_auc)


    def main(self,algorithm,sentiment = False):
        obj = Data("./SMSSpamCollection")
        obj.load_data()
        data = obj.get_data()
        self.algorithm = algorithm 

        if self.algorithm == "SVM":
            message_vector, label = obj.data_convertor(sentiment)
            oversampled_message, oversampeld_label = obj.oversample_data(message_vector, label )
            train_data, test_data = obj.data_split(oversampled_message, oversampeld_label)
            cr,cm,acc,fpr,tpr,roc_auc = self.svm_classifier(train_data,test_data)
            if sentiment:
                print(f"Accuracy with Sentiment Analysis for SVM: {acc}")
                print(cr)
                obj.plot_and_save_confusion_matrix(cm,"SVM with sentiment", "SVM_Sentiment.jpeg")
            else: 
                print(f"Accuracy for SVM: {acc}")
                print(cr)
                obj.plot_and_save_confusion_matrix(cm,"SVM", "SVM.jpeg")
            return cr,cm,acc,fpr,tpr,roc_auc

        elif self.algorithm == "KMeans":
            message_vector, label = obj.data_convertor_tfidf(sentiment)
            
            if sentiment:
                seed = 2300
                cr,cm,acc,fpr,tpr,roc_auc = self.kmeans_clustering(message_vector,label,seed)
                print(f"Accuracy with Sentiment Analysis for KMeans: {acc}")
                print(cr)
                obj.plot_and_save_confusion_matrix(cm,"KMeans with sentiment", "KMeans_Sentiment.jpeg")
            else: 
                seed = 2344 
                cr,cm,acc,fpr,tpr,roc_auc = self.kmeans_clustering(message_vector,label,seed)
                print(f"Accuracy for KMeans: {acc}")
                print(cr)
                obj.plot_and_save_confusion_matrix(cm,"KMeans", "KMeans.jpeg")
            return cr,cm,acc,fpr,tpr,roc_auc


        elif self.algorithm == "CNN": 
            tokenizer = Tokenizer(num_words=5000)
            tokenizer.fit_on_texts(data["Message_after_preprocessing"])
            sequences = tokenizer.texts_to_sequences(data["Message_after_preprocessing"])
            padded_data = pad_sequences(sequences, maxlen=100)

            sentiment_data = data["Sentiment_Analysis"].tolist()
            sentiment_data = np.array(sentiment_data).reshape(-1, 2)
            if sentiment:
                padded_data = np.hstack((padded_data, sentiment_data))

            encoder = LabelEncoder()
            encoder.fit(["ham","spam"])
            required_labels = encoder.transform(data["Label"])
            oversampled_message, oversampeld_label = obj.oversample_data(padded_data,required_labels)
            train_data, test_data = obj.data_split(oversampled_message, oversampeld_label)

            cr,cm,acc,fpr,tpr,roc_auc = self.cnn(train_data,test_data)
            if sentiment:
                print(f"Accuracy with sentiment Analysis for CNN: {acc}")
                print(cr)
                obj.plot_and_save_confusion_matrix(cm,"CNN with sentiment", "CNN_Sentiment.jpeg")
            else: 
                print(f"Accuracy for CNN: {acc}")
                print(cr)
                obj.plot_and_save_confusion_matrix(cm,"CNN", "CNN.jpeg")
            return cr,cm,acc,fpr,tpr,roc_auc
        else: 
            raise ValueError('Invalid Algorithm')


if __name__ == "__main__":
    obj = Model()
    svm_results_sentiment = obj.main("SVM",sentiment=True)
    kmeans_results_sentiment = obj.main("KMeans",sentiment=True)
    cnn_results_sentiment = obj.main("CNN",sentiment=True)
    svm_results_nosentiment = obj.main("SVM",sentiment=False)
    kmeans_results_nosentiment = obj.main("KMeans",sentiment=False)
    cnn_results_nosentiment = obj.main("CNN",sentiment=False)


    # Sentiment Case
    fpr_svm_sentiment, tpr_svm_sentiment, roc_auc_svm_sentiment = svm_results_sentiment[3:]
    fpr_kmeans_sentiment, tpr_kmeans_sentiment, roc_auc_kmeans_sentiment = kmeans_results_sentiment[3:]
    fpr_cnn_sentiment, tpr_cnn_sentiment, roc_auc_cnn_sentiment = cnn_results_sentiment[3:]

    # No Sentiment Case
    fpr_svm_nosentiment, tpr_svm_nosentiment, roc_auc_svm_nosentiment = svm_results_nosentiment[3:]
    fpr_kmeans_nosentiment, tpr_kmeans_nosentiment, roc_auc_kmeans_nosentiment = kmeans_results_nosentiment[3:]
    fpr_cnn_nosentiment, tpr_cnn_nosentiment, roc_auc_cnn_sentiment = cnn_results_nosentiment[3:]

    # Plot ROC Curves for Sentiment Case
    plt.figure(figsize=(10, 6))
    plt.plot(fpr_svm_sentiment, tpr_svm_sentiment, label='SVM (AUC = %0.4f)' % roc_auc_svm_sentiment)
    plt.plot(fpr_kmeans_sentiment, tpr_kmeans_sentiment, label='KMeans (AUC = %0.4f)' % roc_auc_kmeans_sentiment)
    plt.plot(fpr_cnn_sentiment, tpr_cnn_sentiment, label='CNN (AUC = %0.4f)' % roc_auc_cnn_sentiment)

    plt.legend()
    plt.title('ROC Curve - Sentiment Analysis')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.savefig("./output/ROCCurveWithSentimentAnalysis.jpeg")

    plt.figure(figsize=(10, 6))
    plt.plot(fpr_svm_nosentiment, tpr_svm_nosentiment, label='SVM (AUC = %0.4f)' % roc_auc_svm_nosentiment)
    plt.plot(fpr_kmeans_nosentiment, tpr_kmeans_nosentiment, label='KMeans (AUC = %0.4f)' % roc_auc_kmeans_nosentiment)
    plt.plot(fpr_cnn_nosentiment, tpr_cnn_nosentiment, label='CNN (AUC = %0.4f)' % roc_auc_cnn_sentiment)

    plt.legend()
    plt.title('ROC Curve - No Sentiment Analysis')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.savefig("./output/ROCCurveWithoutSentimentAnalysis.jpeg")