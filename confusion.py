import numpy as np
import pandas as pd

class Confusion:
    @staticmethod
    def getMatrix(actual, predict, Display=True):
        if len(actual) < 2:
            raise ValueError('Not enough input arguments. Need at least two vectors as input')
        elif len(actual) != len(predict):
            raise ValueError('Input vectors have different lengths')
        
        un_actual = np.unique(actual)
        un_predict = np.unique(predict)
        
        if len(un_actual) != len(un_predict):
            raise ValueError('Class lists in given inputs are different')
        
        n_class = len(un_actual)
        c_matrix = np.zeros((n_class, n_class))
        class_ref = [f'class{i}==>{class_list[i]}' for i in range(n_class)]
        row_name = [f'Actual_class{i}' for i in range(1, n_class + 1)]
        
        for i in range(n_class):
            for j in range(n_class):
                val = (actual == un_actual[i]) & (predict == un_actual[j])
                c_matrix[i, j] = np.sum(val)
                
        if Display:
            print('Multi-Class Confusion Matrix Output')
            print('Confusion Matrix:')
            print(c_matrix)
        
        Result, RefereceResult = Confusion.getValues(c_matrix)
        
        RefereceResult['Class'] = class_ref
        
        if Display:
            print('Over all values:')
            print(Result)
        
        return c_matrix, Result, RefereceResult
    
    @staticmethod
    def getValues(c_matrix):
        if len(c_matrix.shape) != 2 or c_matrix.shape[0] != c_matrix.shape[1]:
            raise ValueError('Confusion matrix dimension is wrong')
        
        n_class = c_matrix.shape[0]
        TP = np.zeros(n_class)
        FN = np.zeros(n_class)
        FP = np.zeros(n_class)
        TN = np.zeros(n_class)
        
        for i in range(n_class):
            TP[i] = c_matrix[i, i]
            FN[i] = np.sum(c_matrix[i, :]) - c_matrix[i, i]
            FP[i] = np.sum(c_matrix[:, i]) - c_matrix[i, i]
            TN[i] = np.sum(c_matrix) - TP[i] - FP[i] - FN[i]
        
        P = TP + FN
        N = FP + TN
        
        accuracy = (TP + TN) / (P + N)
        Error = 1 - accuracy
        
        AccuracyOfSingle = TP / P
        ErrorOfSingle = 1 - AccuracyOfSingle
        Sensitivity = TP / P
        Specificity = TN / N
        Precision = TP / (TP + FP)
        FPR = 1 - Specificity
        beta = 1
        F1_score = ((1 + beta**2) * (Sensitivity * Precision)) / ((beta**2) * (Precision + Sensitivity))
        MCC = np.maximum(((TP * TN - FP * FN) / ((TP + FP) * P * N * (TN + FN))**0.5), ((FP * FN - TP * TN) / ((TP + FP) * P * N * (TN + FN))**0.5))
        
        pox = np.sum(accuracy)
        Px = np.sum(P)
        TPx = np.sum(TP)
        FPx = np.sum(FP)
        TNx = np.sum(TN)
        FNx = np.sum(FN)
        Nx = np.sum(N)
        pex = ((Px * (TPx + FPx)) + (Nx * (FNx + TNx))) / ((TPx + TNx + FPx + FNx)**2)
        kappa_overall = np.maximum(((pox - pex) / (1 - pex)), ((pex - pox) / (1 - pox)))
        
        po = accuracy
        pe = ((P * (TP + FP)) + (N * (FN + TN))) / ((TP + TN + FP + FN)**2)
        kappa = np.maximum(((po - pe) / (1 - pe)), ((pe - po) / (1 - po)))
        
        Result = {
            'Accuracy': np.sum(np.diag(c_matrix))/np.sum(c_matrix),
            'Sensitivity': np.mean(Sensitivity),
            'Specificity': np.mean(Specificity),
            'Precision': np.mean(Precision),
            'FalsePositiveRate': np.mean(FPR),
            'F1_score': np.mean(F1_score),
            'MatthewsCorrelationCoefficient': np.mean(MCC),
            # 'Kappa': kappa_overall
            
        }
        
        RefereceResult = {
            'AccuracyInTotal': accuracy,
            'ErrorInTotal': Error,
            'Sensitivity': Sensitivity,
            'Specificity': Specificity,
            'Precision': Precision,
            'FalsePositiveRate': FPR,
            'F1_score': F1_score,
            'MatthewsCorrelationCoefficient': MCC,
            'Kappa': kappa,
            'TruePositive': TP,
            'FalsePositive': FP,
            'FalseNegative': FN,
            'TrueNegative': TN
        }
        
        Result = pd.DataFrame.from_dict(Result, orient='index', columns=['Value'])
        RefereceResult = pd.DataFrame(RefereceResult)
        
        return Result, RefereceResult
