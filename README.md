Neural-Networks-and-Machine-Learning
Detecting the transitional state induced by propofol sedation using machine learning algorism


Zhibin Zhou
Department of Cognitive Sciences
University of California, Irvine
Irvine,CA 92617
zhibinz2@uc.edu
		
Abstract
The Abstract paragraph should be indented ½ inch (3 picas) on both left and right-hand margins. Use 10 point type, with a vertical spacing of 11 points. Abstract must be centered, bold and in point size 12. Two line spaces precede the Abstract. The Abstract must be limited to one paragraph.

1	Introduction

1.1	Motivation & specific questions
Propofol is the most commonly used intravenous anesthetic drug today. Millions of people in world take surgeries and minor procedures under propofol anesthesia every day. In larger dosage, propofol induces unconsciousness (drug induced coma) so that major surgeries can be performed. However, larger doses come with higher risks. Larger propofol dose suppress autonomic protective nerve reflects, meaning risk of aspiration, respiratory suppression, hemodynamic instability, hypothermia etc[1]. In lower dosage, it induces a drowsy state so that patients can tolerate most minor procedures without much discomfort, meanwhile with benefits such as autonomic respiratory and protective gag reflex being conserved, hemodynamic better maintained[2].
Accurately measuring the neural correlates of consciousness is a challenge for neuroscience. Studies show that the frontal shifting of alpha oscillations is a prominent feature during propofol anesthesia [3]. The level of alpha activities seems to a determinant feature for different responses to the dosage of anesthesia [4]. Our aim is to develop a robust analytical paradigm to identify the EEG spatial pattern in the transformation of consciousness base on the alpha component during anesthesia. This analytical program can potentially serve as approach for developing better algorism for clinical anesthesia monitoring, enhancing existing EEG monitoring device to help physician titrate anesthetic dosage to achieve the optimal state of sedation.

1.2 	Two types of analysis & why
We used Fast Fourier Transformation to extra alpha component from the original EEG data. Then we combined two types of analysis the PCA (Principle Component Analysis)[5] and LDA (Linear Discrimination Analysis)[6] to source the principal components of the alpha activities and to maximally separate the two states of consciousness, the awake and the drowsy state under anesthesia. PCA and LDA will be integrated together as a Spatial Filter to identify the spatial pattern responsible for the separation of two conscious states.
The reason that PCA and LDA are chosen is because PCA can reduce the dimensionality of the original dataset, increasing interpretability of the principal alpha components and at the same time reducing the effect of noise. And LDA is widely used in pattern recognition and machine learning. It can be employed to find a linear combination of features in the alpha activities that characterize or separate the two the conscious states. The combination of PCA and LDA will serve as a spatial classifier for our purpose
Hypothesis
The spatial distribution of alpha activities are will be different between the awake and the drowsy state. There exists a spatial pattern based on this alpha activities to separate the two states. The pattern will most likely separate the changes of alpha power between the occipital / parietal lobe and the frontal lobe. This spatial pattern can served to identify the state of proper sedation level and to help physician determine the real time dosage of anesthetic.


2	Methods
The experiment & data
Source of the data: 
University of Cambridge Repository [4] 
(https://www.repository.cam.ac.uk/handle/1810/252736
)
Data type: 
Dense array EEG during the awake and drowsy state under propofol anesthesia. A total of 12 subjects were included in this study.
Experimental conditions: 
Subjects were gradually induced into different sedation levels according to plasma propofol concentrations. EEG was recorded during Baseline (the awake state), Mild Sedation, Moderate Sedation (the drowsy state) and Recovery. There are 91 channels. Sapling frequency is 250. Each epoch of data is 7 minutes long, both in the awake and drowsy states.

Time-frequency Analysis
The original EEG data were segmented into 1 second epochs. Fast Fourier Transformation were performed to extra the power of alpha frequency at 11Hz for each epochs. This create two alpha power matrices for both the awake state and the drowsy state over all 91 channels.

Advanced technique
Using Principal Component analysis, the two alpha power matrices can be transformed into 91 principal components. The first ten principal components were selected for further analysis. A   of EEG variables can be transformed into a few Principal components. For Linear Discrimination Analysis, these two alpha power matrices were submitted to our costumed LDA algorithm with a 10-fold cross-validation. We then averaged the 10 models together, since each model is based on 90% of the data in the alpha matrices. The choice of the number of components in PCA and the consistency of the results from our classifier were tested with Akaike information criterion (AIC) and the error rate. These processing steps were implemented using custom MATLAB scripts. Two random subject epochs were selected as test signal. The final results included 12 subjects (6 subjects in both awake and drowsy group).


3	Results
Time-frequency
A total of 2580 epochs (trials) from 12 subjects in both the awake (responsive) and the drowsy group (6 subjects in each state) were extracted. Alpha power matrices for each states were plotted against the 91 channels on heat maps as in Figure 1. More alpha power were seem in the awake (responsive) group, appearing across more channels. The alpha power were not evenly distributed. Significant spatial distribution difference were observed.

Figure 1: Sample Figure Caption

Advanced technique
The choice of number of components is a free parameter. The results of the principal component analysis are shown in Figure 2. The percentage accounted of the total variance for the 91 principal components were plotted. The first 10 to 20 principal components account for most variance. We used AIC to examine the model order. First 10 principal components were tested as shown in Figure 10.

Table 1: Sample table title

Part
Description	
Dendrite	Input terminal
Axon	Output terminal
Soma	Cell Body (contains cell nucleus)

5	Discussion
Summarize results
The results support the hypothesis that a spatial pattern for alpha power exists (Figure 3.)  that can separate the awake state and drowsy state in propofol anesthesia.

Comments on results
The results do support the hypothesis. There is a trade-off between the AIC based on the choice of number of principal components and the error rate based on the LDA classifier as shown in Figure 5. But the spatial pattern of the classifier are very much consistent (Figure 3) for the first 10 principal components with most weight putting on occipital and parietal lobes, and less weight on the frontal lobe.

Future work
In the future, we seek to reduce the error rate in the result. Develop a machine learning paradigm that can detect the achievement of drowsy state as the new data come in, so that it can be applied in clinical EEG monitoring in real time, guiding physician anesthesiologists to determine the optimal dosage of anesthetic for surgical procedures.

Acknowledgments
We thanks the Department of Clinical Neurosciences, Division of Anaesthesia, and Department of Psychology at University of Cambridge for providing this data repository.
References
[1] Illievich UM, Petricek W, Schramm W, Weindlmayr-Goettel M, Czech T, Spiss CK. Electroencephalographic burst suppression by propofol infusion in humans: hemodynamic consequences. Anesth Analg. 1993; 77: 155-160.https://www.ncbi.nlm.nih.gov/pubmed/8317724
[2] Yoon HD, Yoon ES, Dhong ES, Park SH, Han SK, Koo SH, Kim WK. Low-dose propofol infusion for sedation during local anesthesia. Plast Reconstr Surg. 2002; 109: 956-963.https://www.ncbi.nlm.nih.gov/pubmed/11884816
[3] Purdon PL, Pierce ET, Mukamel EA, Prerau MJ, Walsh JL, Wong KF, Salazar-Gomez AF, Harrell PG, Sampson AL, Cimenser A, Ching S, Kopell NJ, Tavares-Stoeckel C, Habeeb K, Merhar R, Brown EN. Electroencephalogram signatures of loss and recovery of consciousness from propofol. Proceedings of the National Academy of Sciences of the United States of America. 2013; 110: E1142-1151.https://www.ncbi.nlm.nih.gov/pubmed/23487781
[4] Chennu S, O'Connor S, Adapa R, Menon DK, Bekinschtein TA. Brain Connectivity Dissociates Responsiveness from Drug Exposure during Propofol-Induced Transitions of Consciousness. PLoS Comput Biol. 2016; 12: e1004669.https://www.ncbi.nlm.nih.gov/pubmed/26764466
[5] Jao PK, Chavarriaga R, Millan JDR. Using Robust Principal Component Analysis to Reduce EEG Intra-Trial Variability. Conf Proc IEEE Eng Med Biol Soc. 2018; 2018: 1956-1959.https://www.ncbi.nlm.nih.gov/pubmed/30440781
[6] Fu R, Tian Y, Bao T, Meng Z, Shi P. Improvement Motor Imagery EEG Classification Based on Regularized Linear Discriminant Analysis. J Med Syst. 2019; 43: 169.https://www.ncbi.nlm.nih.gov/pubmed/31062175

