################################################################################# 
#                                                                               #
#  CLASIFICA_MAIN                                                               #
#                                                                               #
#      Toma una muestra aleatoria de DT (dataframe Content|File|Tag|id), que    #
#      contiene los 19997 documentos de texto, para entrenar un modelo          #
#      de clasificacion de textos en los 20 grupos de "Tag".                    #
#                                                                               #
#      DT se carga dentro de create_DT_sample.R de "./data/DT.RData" pero es    #
#      posible generarlo de nuevo con el script built_DT.R. En este caso hay    #
#      que adaptar los nombres de las variables de DT porque se llaman          #
#      de manera distinta.                                                      #
#                                                                               #
#      Este script tiene, al principio, cuatro parametros de ejecucion para     #
#      controlar:                                                               #
#                                                                               #
#      NEW_DATASETS -- Genera nuevo o carga ultimo DT_sample de DT. TRUE/FALSE  #
#                                                                               #
#      SVM_CLASSIF      -- Entrena modelo "support vector machines" libreria    #  
#                          "e1071".                                             #            
#      KSVM_CLASSIF     -- Entrena modelo "support vector machines" libreria    # 
#                          "kernlab" con kernel propio.                         #
#      LOGISTIC_CLASSIF -- Entrena modelo logistic regression one vs. all.      #
#                                                                               #
#                                                                               #
#  Autores:     Maria Calvo                                                     #
#               David Grinan                                                    #
#               Jordi Aceiton                                                   #
#                                                                               #
#  Fecha:       22/05/2017                                                      #
#                                                                               #
#################################################################################

## clear console and workspace
cat("\014")
rm(list= ls())

## FUNCIONES
source("./functionScripts/create_DT_sample.R")
source("./functionScripts/process_texts.R")
source("./functionScripts/X_y_dataset.R")
source("./functionScripts/train_test.R")

#=================================================================================
# PARAMETROS DE EJECUCION:

# Datos:
NEW_DATASETS <- TRUE   # Nuevos DT_sample, Xtrain, Xtest, ytrain e ytest 

## Metodos de clasificacion:
SVM_CLASSIF= FALSE       # Por "support vector machines" libreria "e1071".
KSVM_CLASSIF= FALSE      # Por "support vector machines" libreria "kernlab".
LOGISTIC_CLASSIF= FALSE  # Por logistic regression -"one vs all".
#=================================================================================

#=================================================================================
# GENERACION DEL DATASET X e y -- TRAIN, SET.
#=================================================================================

if(NEW_DATASETS) {
        # Muestra aleatoria de documentos de DT.
        # Sampleado uniforme.
        DT_sample <- create_DT_sample(num_docs= 5000,    # numero de documentos.
                                      seed= 1234)        # semilla.
        
        
        # Procesado de los textos de DT_sample.
        corpus <- process_texts(DT_sample[,Content],    # vector de textos.
                                stem= TRUE)             # lematizacion (stem).
        
        save(corpus, file= "./data/corpus.RData")
        
        # Se genera el dataset: matriz documentos-terminos X y vector etiquetas
        # por pertenencia de documentos en grupos, y.
        Xy <- X_y_dataset(corpus, 
                          labels= DT_sample$Tag, 
                          word.length= c(2,15),   # Solo palabras de c(min.letras, max.letras)
                          freq.bounds= c(1,Inf),  # intervalo para ocurrencia de terminos.
                          tfidf= TRUE,            # matriz doc-term con o sin idf.
                          with.normalize= TRUE,   # matriz tfidf normalizada? 
                          with.sparse= TRUE, 
                          sparse= 0.995)          # valor de sparse por termino. Se quitan 
                                                  # terminos con sparse mayor.

        # Se parte la matriz de documentos-terminos "X" y vector "y" en train y test sets.
        sets <- train_test(X= Xy$X,
                           y= Xy$y,
                           train.part= 0.8)      # proporcion de X,y para Xtrain,ytrain        
        
        # Preparacion de las piezas para entrenar el modelo y clasificar.
        Xtrain <- sets$train$X
        ytrain <- sets$train$y
        
        idf <- Xy$idf
        
        Xtest  <- sets$test$X
        ytest  <- sets$test$y
        
        save(Xtrain, ytrain, idf, Xtest, ytest, file= "./data/train_test.RData")
        
        # Limpieza del workspace.
        rm(corpus, sets, Xy)
        rm(create_DT_sample, process_texts, X_y_dataset, train_test)

} else {
        # Cargamos datos.
        cat("\nCargando datos...")
        load("./data/DT_sample.RData")
        load("./data/train_test.RData")
        cat(" ok! \n\n")
}
#################################################################################
#################################################################################


#=================================================================================
# CLASIFICACION POR  "SUPORT VECTOR MACHINES" library(e1071).
#=================================================================================
if(SVM_CLASSIF) {
        
        library(e1071) 
        
        ## ENTRENAMIENTO DEL MODELO POR SVM:
        cat("\ne1071 - Support Vector Machines. Entrenando modelo...")
        
        # Para que haga svm clasificacion, type= C-classification. 
        #
        # Hay 4 kernels disponibles:
        #       linear: u'*v,  polynomial: (gamma*u'*v + coef0)^degree
        #       radial: exp(-gamma*|u-v|^2), sigmoid: tanh(gamma*u'*v+coef0) 
        
        svm_model <- svm(Xtrain, ytrain, 
                         type= "C-classification",
                         probability= TRUE, 
                         kernel= "radial",
                         cost= 1)  # Parametro de regularizacion (1 por defecto).
        cat(" ok!\n")
        
        ## ACCURACIES DEL MODELO POR SVM:
        cat("e1071 - Support Vector Machines. Accuracies...")
        pred_train <- as.numeric(predict(svm_model, Xtrain))
        pred_test <- as.numeric(predict(svm_model, Xtest))
        
        accu_svm_train <-  signif(sum(pred_train==ytrain)/length(ytrain), digits= 3) 
        accu_svm_test <-  signif(sum(pred_test==ytest)/length(ytest), digits= 3)
        cat(" ok!\n")
}

#=================================================================================
# CLASIFICACION POR  "SUPORT VECTOR MACHINES" library(kernlab). Coseno Kernel
#=================================================================================
if(KSVM_CLASSIF) {
        
        library(kernlab)
        
        ## ENTRENAMIENTO DEL MODELO POR SVM:
        cat("\nkernlab - Support Vector Machines. Entrenando modelo...")
        
        # kernels: Cualquiera de la libreria. Por defecto: "rbfdot" (Radial Basis 
        #          kernel "Gaussian") o kernel definido por el usuario.
        
        # Nucleo basado en la similaridad por coseno.
        coseno_kern <- function(x,y) {sum(x*y)/sqrt(sum(x^2)*sum(y^2))}
        class(coseno_kern) <- "kernel"
        
        # SVM:
        ksvm_model <- ksvm(Xtrain, ytrain, 
                           type= "C-svc", 
                           kernel= coseno_kern)
        cat(" ok!\n")
        
        ## ACCURACIES DEL MODELO POR SVM:
        cat("\nkernlab - Support Vector Machines. Accuracies...")
        pred_train <- as.numeric(predict(svm_model, Xtrain))
        pred_test <- as.numeric(predict(svm_model, Xtest))
        
        accu_svm_train <-  signif(sum(pred_train==ytrain)/length(ytrain), digits= 3) 
        accu_svm_test <-  signif(sum(pred_test==ytest)/length(ytest), digits= 3)
        cat(" ok!\n")
}


#=================================================================================
# CLASIFICACION POR  "ONE vs ALL".
#=================================================================================
if(LOGISTIC_CLASSIF) {
        source("./functionScripts/logist_ml.R")
        source("./functionScripts/predict.logist.R")
        
        ## ENTRENAMIENTO DEL MODELO POR ONE vs ALL:
        cat("\n")
        logist_model <- logist_ml(Xtrain, ytrain,
                                  maxit= 50,         # max iterations to optim{stats}
                                  lambda= 1)         # Parametro de regularizacion.
        
        ## ACCURACIES DEL MODELO POR LOGISTIC REGRESSION:
        cat("\nOne vs all - logistic regression. Accuracies...")
        pred_train <- predict.logist(logist_model$weights, Xtrain)
        pred_test <- predict.logist(logist_model$weights, Xtest)
        
        accu_logist_train <-  signif(sum(pred_train$prediction == ytrain)/length(ytrain), digits= 3)
        accu_logist_test <-  signif(sum(pred_test$prediction == ytest)/length(ytest), digits= 3)
        
        #pred_probs_train <- pred_train$class_h
        #pred_probs_test <- pred_test$class_h
        
        cat(" ok!\n")
        
        # Matriz de aciertos.
        #confMat_logist <- table(pred_test, ytest)
        rm(costFunctionReg, JcostFunctionReg, gradientReg, logist_ml, predict.logist)
}


rm(NEW_DATASETS, LOGISTIC_CLASSIF, SVM_CLASSIF, KSVM_CLASSIF)

