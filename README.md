# PJA-ASI-14C-GR4

## Contributors
- Cezary Sieczkowski s22595
- Konrad Szwarc s23087
- Patryk Polnik s22515
- Tomasz Iwanowski s18438

## Instrukcja uruchomienia środowiska
1. Download and install [Miniconda](https://docs.anaconda.com/free/miniconda/index.html)
2. Download file `environment.yml` from catalog `env`
3. In terminal use command `conda env create -f environment.yml`
4. Activate created environment using command `conda activate ASI`

<h2>Crab age prediction model</h2>

<p>The model here created will be able to predict crab age based on given biometrics and sex. Model will take as input features listed below:</p>

<ul>
    <li>Sex -> String; Values=['I','M','F']</li>
    <li>Length -> float</li>
    <li>Diameter -> float</li>
    <li>Height -> float</li>
    <li>Weight -> float</li>
    <li>Shucked Weight -> float</li>
    <li>Viscera Weight -> float</li>
    <li>Shell Weight -> float</li>
</ul>

<p>The target will be Age value that type is int. We will use Random forest model.</p>

<p>For data cleaning look up file raw_pipeline\data_analysis.ipynb</p>

<p>For model evaluation look up file raw_pipeline\model_eval.ipynb</p>
