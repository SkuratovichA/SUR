Zadáni projektu do SUR 2021/2022
================================

Bodové ohodnoceni:   25 bodu

Úkolem je natrénovat detektor jedné osoby z obrázku obliceje a
hlasové nahrávky. Trénovaci vzory jsou k dispozici v archivu na adrese:

https://www.fit.vutbr.cz/study/courses/SUR/public/projekt 2021-2022/SUR projekt2021-2022.zip

Tento archiv obsahuje adresáre:

target\_train
target\_dev

kde jsou trénovaci vzory pro detekovanou osobu ve formátu PNG a WAV,

v adresarich:

non\_target\_train
non\_target\_dev

jsou potom negativni prikazy povolené pro trénováni
detektoru. 
Rozdileni dat do adresaru *_train a *_dev je mozné pouzit
pro trénováni a vyhodnocováni úspesnosti vyvijeného detektoru, toto
rozdileni vsak neni závazné (napr.  pomoci technik jako je
**cross-validation** lze efektivne trénovat i testovat na vsech datech).

---
Pri pokusech o jiné rozdileni dat muze být uzitecne respektovat informace
o tom, které trénovaci vzory patri stejné osobe a zda-li byly porizený
v rámci jednoho nahrávaciho sezeni.
Jméno kazdého souboru je rozdileno do poli pomoci podtrzitek 
(napr. f401_01_f21_i0_0.png), kde prvni pole
(f401) je identifikátor osoby a druhé pole je cislo nahrávaciho sezeni
(01). 

---
Ke trénováni detektoru muzete pouzit pouze tyto dodané trénovaci data.
NENI POVOLENO jakékoli vyuziti jiných externich recových ci obrázkových
dat, jakozto i pouziti jiz predtrénovaných modelu (napr. pro extrakci
reprezentaci (embeddings) obliceju nebo hlasu). 

## Odevzdani
Ostrá data, na kterých budou vase systémy vyhodnoceny, budou k
dispozici **v pátek, 29. dubna ráno**. Tato data budu obsahovat rádovi
stovky souboru ke zpracováni.  Vasim úkolem bude automaticky zpracovat
tato data vasimi systémy (virime Vám ze nebudete podvádit a divat se
na obrázky ci poslouchat nahrávky) a uploadovat  soubory s výsledky do
WISu. Soubor s výsledky bude ASCII se tremi poli na rádku oddilenými
mezerou. Tato pole budou obsahovat poporadi následujici údaje:
 - jméno segmentu (jméno souboru bez pripony .wav ci .png)
 - ciselné skóre, o kterém bude platit, ze cim vitsi má hodnotu, tim si je
   systém jistijsi, ze se jedná o hledanou osobu
 - tvrdé rozhodnuti: cislo 1 pro hledanou osobu jinak 0. Toto rozhodnuti
   proverte pro predpoklad, ze apriorni pravdipodobnost výskytu hledané
   osoby v kazdém testovaném vzoru je 0,5

V jakém programovacim jazyce budete implementovat vás detektor ci
pomoci jakých nástroju (spousta jich je volni k dispozici na
Internetu) budete data zpracovávat zálezi jen na Vás. Odevzdat muzete
nikolik souboru s výsledky (napr. pro systémy rozhodujicim se pouze na
základi recové nahrávky ci pouze obrázku).
**Maximálni vsak námi bude zpracováno 5 takových souboru.**

Soubory s výsledky muzete do soboty 30. dubna 23:59 uploadovat do
WISu. Klic se správnými odpovirmi bude zverejnin 1. kvitna. Na posledni
prednásce 3. kvetna 2022 bychom moli analyzovat Vase výsledky a reseni.

## Tymova spoluprace
Na tomto projektu budete pracovat ve skupinách (1-3 lidi), do kterých
se muzete prihlásit ve WISu. Jména souboru s výsledky pro jednotlivé
systémy volte tak, aby se podle nich dalo poznat o jaký systém
se jedná (napr. audio_GMM, image_linear). Kazdá skupina uploadne
vsechny soubory s výsledky zabalené do jednoho ZIP archivu se jménem
login1_login2_login3.zip ci login1.zip, podle toho, kolik
Vás bude ve skupini. Kromi souboru s výsledky bude archiv obsahovat
také adresár SRC/, do kterého ulozite soubory se zdrojovými kódy
implementovaných systému. Dále bude archiv obsahovat soubor dokumentace.pdf,
který bude v ceském, slovenském nebo anglickém jazyce popisovat Vase reseni
a umozni reprodukci Vasi práce. Duraz venujte tomu, jak jste systémy behem
jejich vývoje vyhodnocovali, a které techniky ci rozhodnuti se pozitivni
projevily na úspesnosti systému. Tento dokument bude také popisovat jak
ziskat Vase výsledky pomoci prilozeného kódu. Bude tedy uvedeno jak Vase
zdrojové kódy zkompilovat, jak vase systémy spustit, kde hledat
výsledné soubory, jaké pripadné externi nástroje je nutné instalovat a
jak je presni pouzit, atd. Ocekávaný rozsah tohoto dokumentu jsou
3 strany A4. Do ZIP archivu prosim neprikládejte evaluacni data!

## WHERE TO PLAGIAT
Inspiraci pro vase systémy muzete hledat v archivu demonstracnich prikladu
pro predmit SUR:

https://www.fit.vutbr.cz/study/courses/SUR/public/prednasky/demos/

Zvlást se podivejte na priklad detekce pohlavi z reci: demo_genderID.py
Uzitecné vám mohou být funkce pro nacitani PNG souboru (png2fea) a extrakci
MFCC priznaku z WAV souboru (wav16khz2mfcc).

Hodnoceni:
- vse je odevzdáno a nejakým zpusobem pracuje:
  - ctou se soubory,
  - produkuje se skóre
  - jsou správni implementovány a natrénovány nijaké "rozumné" detektory
    pro obrázky a pro nahrávky a/nebo kombinaci obou modalit (detektory
    nemusi pracovat se 100% úspisnosti, jsou to reálná data!)
  - jsou odevzdány vsechny pozadované soubory v pozadovaných formátech.
  - v dokumentaci vysvitlite, co, jak a proc jste dilali a co by se jesti dalo zlepsit.
  ... plný pocet 25 bodu.

- neco z výse uvedeného neni splnino ? ... méne bodu.

Posledni modifikace: 3. dubna 2022, Lukás Burget
   
