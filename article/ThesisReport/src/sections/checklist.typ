#import "../callouts.typ":icon
#set text(size: 0.9em)


= Checklist #icon("tick")/#icon("cross")

== Report Checklist
This checklist was created by Chang Sun for Bachelor's and Master's students at the Department of Advanced Computing Sciences, Maastricht University. It supports writing a complete, clear, and scientifically sound thesis. Students are encouraged to share this checklist with supervisors and highlight the items they have addressed.

// #heading[1][Introduction]
=== Introduction
+ *Problem Statement*: Clearly define the key problem your thesis addresses. #link(<check_research_question>,"research_question") #icon("tick")
+ *Significance*: Explain why this problem is important and urgent.#link(<check_significance>,"significance") #icon("tick")
+ *Research Challenges*: Identify why the problem has not been solved yet.#icon("warning")
+ *Brief Description of This Work*: Summarize your approach and how it differs from existing methods. #link(<check_brief_description>,"desc") #icon("tick")
+ *(Optional)* *Illustrative Diagram*: Include a simple, easy-to-understand diagram.#icon("cross")
+ *Key Contributions*: List and highlight the innovative contributions.#link(<check_contribution>,"contribution") #icon("tick")
+ *Thesis Structure*: Briefly describe what each chapter covers.#icon("cross")

// #heading[1][Related Work]

=== Related Work
+ Summarize state-of-the-art methods relevant to your thesis.
+ Discuss strengths and limitations.
+ Identify gaps in current work.
+ Explain how your thesis addresses these gaps.

// #heading[1][Methods / Methodology]

=== Methodology
+ *Approach Overview*: High-level view with a well-labeled diagram #icon("cross")
+ *Components*: Explain each component in separate subsections.
+ *Implementation Details*:
  + Method implementation
  + Architecture choices, algorithms, and software
  + Code must be open-source and documented on GitHub.

// #heading[1][Datasets]

=== Datasets
+ *Data Sources & Collection*:
  + What, where, when, and how data was collected #icon("cross")
  + Flowchart: source → filtering → final dataset #icon("cross")
  + Inclusion/exclusion criteria and counts
  + Provide access details in footnotes or appendix
+ *Dataset Description*:
  + Size, distributions, missing values
  + Preprocessing steps

// #heading[1][Experiments]

=== Experiments
+ *Design*:
  + Dataset splits (training/test/validation)
  + Which experiments are run, which parameters tested
+ *Settings*:
  + Hardware (GPU/CPU model)
  + Epochs, batch size, learning rate, etc.
+ *(Include figures/flowcharts when design is complex.)*

// #heading[1][Evaluation Methods]

=== Evaluation Methodology
+ Describe performance evaluation strategy.
+ List and define metrics (e.g., accuracy, precision).
+ Include equations and implementation notes if possible.

// #heading[1][Results]

=== Results
+ Use tables and figures to report results clearly.
+ Highlight key numbers readers should focus on.
+ *Statistical Significance*:
  + Show deviations, error bars, or other significance metrics.
+ Objectively compare models, parameters—avoid interpretations here.

// #heading[1][Discussion]

=== Discussion
+ Interpret findings and patterns.
+ Reflect on unexpected trends.
+ *Limitations*: What didn’t work, assumptions, robustness.
+ *Future Work*: Propose directions for improvement.
+ *(Optional)* Discuss scientific and societal impacts.

// #heading[1][Conclusion]
=== Conclusion
+ Concise summary of your thesis.
+ Highlight discoveries and findings.
+ Share lessons learned.
+ *(If not in Discussion)* mention future work directions.

// #heading[1][References]
=== References
+ All references must include DOI or persistent IDs.
+ Prefer peer-reviewed papers. Preprints (e.g., arXiv) are allowed but disclose their proportion.

// #heading[1][Note on Chapter Organization]
=== Chapter Organization
Sections 4–8 may be structured in one of the following ways:

+ Separate each as an individual chapter.
+ Combine:
  + *Datasets + Experiments + Evaluation* in one chapter;
  + *Results + Discussion* in another.
+ Or:
  + *Dataset* in one chapter;
  + *Experiments, Evaluation, Results* in another;
  + *Discussion* separately.
+ Other logical groupings are allowed if all parts are clearly covered.

// #heading[1][Defense Presentation Checklist]
== Defense Presentation Checklist
+ What is the problem?
+ Why is it significant?
+ Related work
+ Research questions
+ Datasets, study design, and experiments
+ Results and how they answer the research questions
+ List of key contributions
+ Future work directions
