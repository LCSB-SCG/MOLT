# 1. Lack of Comparative Benchmarking

    -  R1 : "Paper lacks any comparison to other tracking algorithms; either TBD or E2E despite many of the cited related works referencing many software libraries which can perform this task. Without comparison readers have no ability to understand how well this approach works in comparison to what already exists. Without comparing against standard baselines (e.g., TrackMate, MaMuT, or Cell Tracking Challenge leaders), it is impossible to know if MOLT’s performance (IDF1 ~99%) is actually good for these specific datasets or if the datasets are simply trivial."
    - R1 : "While broadly lacking any E2E algorithm comparisons, there is a stark absence of comparison between specialist models (like the proposed work) and generalist models like SAM3."
    - R2: "I did not find it clear how the ability of this algorithm to work with multiple types of images is specifically a result of the proposed method. ... To clarify the authors’ claims, it would be helpful to have a quantitative demonstration of a single algorithm, e.g. reference 21, performing well on S1, but poorly on S2, and vice versa with another referenced algorithm, and a comparison to the performance of this paper’s algorithm on both tasks."
    - R2: "Quantitative comparisons should use a consistent set of metrics comparing this paper’s algorithm performance to that of the other algorithms mentioned in this paper on identical data sets."
    - R2: "I would also recommend comparing the performance of this algorithm to those used in TrackMate..."
    - R4: "For improved context and transparency within the Tracking-by-Detection (TBD) paradigm, the authors should consider including in Table 1 a comparison against a standard, training-free TBD baseline (e.g., Hungarian linking or TrackMate), instead of restricting the comparison to internal variants."

    We addressed the reviewers comments by adding comparision to three commonly used tracking algorithms from trackmate. We additionally used the cell tracking challenge metrics for a better comparison to different benchmarks (however it is not directly comparable to the official results in the cell tracking challenge as they use the results on test datasets where the ground truth is not publicly available, we evaluated our methods on the train datasets with available ground truth and verified that none of the data has been used in the cellopse-sam model to mitigate data leakage.)

# 2. Evaluation Data and Realism

    - R1 : "The evaluation presented in the paper leverages silver ground truth annotations from 3 other publications. These annotations represent a clean and high quality mask to work from. Tracking by detection needs to handle segmentation failures upstream of the tracker (i.e. jitter, fragmentation, missed detections, spurious detections), something which is missing from the data used to evaluate the method. The basically perfect detection masks being used to evaluate the method only proves that the graph based logic for identity collision resolution is correct for the studied cases. Its uncertain how well the tool would track real world data with upstream segmentation errors."
    - R1 : "Measuring tracking performance against actual segmentation algorithm output instead of ground truth masks would measure this capability."
    - R2: "The authors may consider swapping their IDF1 metric for any of the AMI, NMI, P@1, RP, or MAP@R metrics used in reference 11. Similarly, the authors could benchmark performance of their algorithm on Fluo-N3DH-SIM+, which is used in reference 11, instead of Fluo-N2DH-GOWT1."
    - R3: "You explicitly use binarized ground-truth segmentations as input and note that false positives do not exist in this evaluation. This is fine, but it must be stated as a limitation. If possible, add one experiment using automated segmentations to demonstrate robustness to realistic errors."
    - R4: "The evaluation uses binarized ground truth segmentations, which eliminates segmentation noise inherent in real-world data. Please explicitly discuss this bias and how imperfect input segmentations would affect the 'Union Map' and conflict resolution steps."

    All of the reviewers comments have been addressed, by adding evaluations on automated segmentations from cellpose-sam. A general purpose cell segmentation model. The tracking results are also still available on Silver ground truth for all the algorithms and use TRA celltracking challenge measure for better comparision.

# 3. Methodological Justification and Clarity

    - R1 : "Affine registration of the frames is performed against frame0, instead of maintaining a difference from the prior time frame. The tracking logic could be more robust if registration occurs between each pair of frames which would allow for pairwise union maps, but without the increasing distortion from frame0 introducing additional error. Evaluating the difference in performance between those two approaches to justify the selection the algorithm made would support the paper."
    - R1 : "Authors should defend the selection of 'morphological event detection' over more traditional detection accuracy metrics like those used in the cited Cell Tracking Challenge."
    - R2: "In the background section, it’s not clear to me how the discussion of TBD vs. E2E algorithms contributes to the understanding of the importance of this paper’s technique. E2E algorithms are criticized for being domain specific. Is it not possible to re-train these neural networks on other image types? Are they truly fixed in their domain, or just inconvenient to use? Is the goal of this paper to demonstrate that this algorithm is easier to use and more flexible than neural network E2E algorithms?"
    - R2: "Table 1 compares affine and deformable registrations, but there is limited discussion of deformable registration before this, and no discussion of deformable registration in the methods. Given this, and that the deformable registration categorically performs worse than the affine registration according to this table, I would drop deformable registration from the manuscript entirely."
    - R3: "You use Greedy registration computed on raw microscopy images and apply transforms to segmentations with nearest-neighbor interpolation, and you state performance depends strongly on registration accuracy. Please report the key registration settings."
    - R4: "Please explicitly describe potential failure modes of the registration step and how registration errors propagate to downstream lineage tracking, as the method’s performance depends on the accuracy of the underlying registration algorithm."

# 4. Metrics and Statistical Reporting

    - R1 : "S1 tracking results are qualitative only. Preparing a small manually validated dataset for this evaluation is needed to move beyond a visual inspection spot check."
    - R2: "The lower end of the range of the accuracy percentages and IDF1 scores mentioned in 'Results on S2' do not appear in Table 1, despite the text stating that they do. Please update."
    - R4: "Please report the standard deviation or variance alongside the mean values for Accuracy and IDF1 in Table 1 to demonstrate the stability of the method across different datasets or ROIs."
    - R4: "While the manuscript defines positive and negative cases for the 'morphological detection accuracy' metric, it does not explicitly state how these quantities are combined into a final accuracy. Please provide a formula or equation to clarify this and to support an accurate interpretation of the reported values."

# 5. Clarity and Presentation of Results

    - R1 : "Given Figure 4, once affine alignment is done the tracking objects appear to be very well separated, indicating a trivial tracking problem that many algorithms would be able to solve."
    - R2: "It’s hard to follow the difference between the linage and tracked images included in 'visuals.zip'. It would helpful if the authors could identify what information the reader is supposed to glean from each animation and perhaps include a section of the video where we zoom in on a single cell dividing over time and watch how the lineage vs. track images change over a short period of time in a small ROI. It would also help if we could see these images as part of a tryptic: the original image, the lineage image, and the track image all next to one another and moving together in a single video."
    - R3: "The strongest technical contribution is the graph-based conflict resolution that maps conflicting union associations to unique lineage IDs (connected groups), addressing splits caused by registration and nearest-neighbor interpolation. Make this the headline novelty and distinguish it from earlier overlap-based trackers."
    - R4: "Rewrite the Abstract and Conclusion to explicitly state the limitation that the method cannot handle object disappearance or reappearance in Scenario 2, as the current claims of robustness are overstated."

# 6. Reproducibility and Data Availability

    - R3: "You state cell-tracking data comes from the Cell Tracking Challenge and the beta-amyloid dataset will be available within 12 months. Consider whether you can release a subset or derived artifacts earlier to support reproducibility expectations."
    - R4: "The paper defines positive as no event and negative as event present, and true/false based on matching the number of children between prediction and ground truth. This is workable, but the naming is easy to misread. Consider renaming or adding a short explanation up front."

    
