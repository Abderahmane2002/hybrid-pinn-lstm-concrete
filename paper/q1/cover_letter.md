[Date]

Dear Editor,

Please find enclosed our manuscript "Learning moisture transport parameters of blended concretes from drying curves: identifiability, generalisation across binders, and a comparison of physics-embedded and physics-informed neural networks", submitted as a research article to the *Journal of Building Engineering*.

Moisture transport parameters of concrete are usually identified from drying or absorption curves one mix at a time, and machine-learning models are increasingly proposed to predict them from mix composition. Using the drying database of Zhuang et al. (J. Build. Eng. 53 (2022) 104602) and the transport model of Chen et al. (J. Build. Eng. 45 (2022) 103640), we examine what such models can and cannot learn from gravimetric data.

The main findings are:

- Drying curves identify two parameters per mix. The reference diffusivity is precise; the permeability is uncertain to a few tenths of a decade and depends on fixed constitutive constants. We quantify both with profile likelihoods, synthetic recovery tests and a sensitivity study.
- A physics-embedded network trained through the transport solver and classical inversion followed by linear regression predict withheld mixes equally well (R² = 0.9931 and 0.9934). When a whole binder family is withheld, only models that combine the transport model with a simple composition map remain accurate.
- End-to-end training is more robust when the drying curves are short.
- A physics-informed neural network fits the same data with permeabilities about one order of magnitude too low. Diagnostic runs show that this is a property of its soft-constrained loss.

We believe the work will interest readers concerned with durability modelling and with the use of machine learning for building materials. The reconstructed data and the code that reproduces every result are publicly available.

This manuscript has not been published and is not under consideration elsewhere. All authors have approved the submission and declare no competing interests.

Yours sincerely,

Xuande Chen, on behalf of all authors

Department of Mathematics, Computer Science and Engineering, Université du Québec à Rimouski (UQAR), Rimouski, QC, Canada

[e-mail]
