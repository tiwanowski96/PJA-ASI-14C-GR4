# PJA-ASI-14C-GR4

## Contributors
- Cezary Sieczkowski s22595
- Konrad Szwarc s23087
- Patryk Polnik s22515
- Tomasz Iwanowski s18438

## Instrukcja uruchomienia środowiska
1. Pobierz i zainstaluj [Miniconda](https://docs.anaconda.com/free/miniconda/index.html)
2. Pobierz plik `environment.yml` z katalogu `env`
3. W terminalu wykonaj polecenie `conda env create -f environment.yml`
4. Uruchom utworzone środowisko wykonując polecenie `conda activate ASI`

<h2>Model przewidujący wiek kraba</h2>

<p>Stworzony model ma za zadnie przewidzieć wiek kraba na bazie danych biometrycznych oraz płci. Poniżej znajduje się lista cech znajduących się w bazowych danych i krótki opis:</p>

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

<p>Naszym celem będzie przewidzić wartość Age, która jest typem intger. Użyjemy do tego modelu Random forest.</p>

<p>Jeśli chcesz sprawdzić jak zostało przeprowadzone pierwotne czyszczenie danych sprawdź model_origin\data_analysis.ipynb</p>

<p>Jeśli chcesz sprawdzić pierwotną ewaluacje modeli sprawdź model_origin\model_eval.ipynb</p>
