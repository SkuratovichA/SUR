Zad�ni projektu do SUR 2021/2022
================================

Bodov� ohodnoceni:   25 bodu

�kolem je natr�novat detektor jedn� osoby z obr�zku obliceje a
hlasov� nahr�vky. Tr�novaci vzory jsou k dispozici v archivu na adrese:

https://www.fit.vutbr.cz/study/courses/SUR/public/projekt 2021-2022/SUR projekt2021-2022.zip

Tento archiv obsahuje adres�re:

target\_train
target\_dev

kde jsou tr�novaci vzory pro detekovanou osobu ve form�tu PNG a WAV,

v adresarich:

non\_target\_train
non\_target\_dev

jsou potom negativni prikazy povolen� pro tr�nov�ni
detektoru. 
Rozdileni dat do adresaru *_train a *_dev je mozn� pouzit
pro tr�nov�ni a vyhodnocov�ni �spesnosti vyvijen�ho detektoru, toto
rozdileni vsak neni z�vazn� (napr.  pomoci technik jako je
**cross-validation** lze efektivne tr�novat i testovat na vsech datech).

---
Pri pokusech o jin� rozdileni dat muze b�t uzitecne respektovat informace
o tom, kter� tr�novaci vzory patri stejn� osobe a zda-li byly porizen�
v r�mci jednoho nahr�vaciho sezeni.
Jm�no kazd�ho souboru je rozdileno do poli pomoci podtrzitek 
(napr. f401_01_f21_i0_0.png), kde prvni pole
(f401) je identifik�tor osoby a druh� pole je cislo nahr�vaciho sezeni
(01). 

---
Ke tr�nov�ni detektoru muzete pouzit pouze tyto dodan� tr�novaci data.
NENI POVOLENO jak�koli vyuziti jin�ch externich recov�ch ci obr�zkov�ch
dat, jakozto i pouziti jiz predtr�novan�ch modelu (napr. pro extrakci
reprezentaci (embeddings) obliceju nebo hlasu). 

## Odevzdani
Ostr� data, na kter�ch budou vase syst�my vyhodnoceny, budou k
dispozici **v p�tek, 29. dubna r�no**. Tato data budu obsahovat r�dovi
stovky souboru ke zpracov�ni.  Vasim �kolem bude automaticky zpracovat
tato data vasimi syst�my (virime V�m ze nebudete podv�dit a divat se
na obr�zky ci poslouchat nahr�vky) a uploadovat  soubory s v�sledky do
WISu. Soubor s v�sledky bude ASCII se tremi poli na r�dku oddilen�mi
mezerou. Tato pole budou obsahovat poporadi n�sledujici �daje:
 - jm�no segmentu (jm�no souboru bez pripony .wav ci .png)
 - ciseln� sk�re, o kter�m bude platit, ze cim vitsi m� hodnotu, tim si je
   syst�m jistijsi, ze se jedn� o hledanou osobu
 - tvrd� rozhodnuti: cislo 1 pro hledanou osobu jinak 0. Toto rozhodnuti
   proverte pro predpoklad, ze apriorni pravdipodobnost v�skytu hledan�
   osoby v kazd�m testovan�m vzoru je 0,5

V jak�m programovacim jazyce budete implementovat v�s detektor ci
pomoci jak�ch n�stroju (spousta jich je volni k dispozici na
Internetu) budete data zpracov�vat z�lezi jen na V�s. Odevzdat muzete
nikolik souboru s v�sledky (napr. pro syst�my rozhodujicim se pouze na
z�kladi recov� nahr�vky ci pouze obr�zku).
**Maxim�lni vsak n�mi bude zpracov�no 5 takov�ch souboru.**

Soubory s v�sledky muzete do soboty 30. dubna 23:59 uploadovat do
WISu. Klic se spr�vn�mi odpovirmi bude zverejnin 1. kvitna. Na posledni
predn�sce 3. kvetna 2022 bychom moli analyzovat Vase v�sledky a reseni.

## Tymova spoluprace
Na tomto projektu budete pracovat ve skupin�ch (1-3 lidi), do kter�ch
se muzete prihl�sit ve WISu. Jm�na souboru s v�sledky pro jednotliv�
syst�my volte tak, aby se podle nich dalo poznat o jak� syst�m
se jedn� (napr. audio_GMM, image_linear). Kazd� skupina uploadne
vsechny soubory s v�sledky zabalen� do jednoho ZIP archivu se jm�nem
login1_login2_login3.zip ci login1.zip, podle toho, kolik
V�s bude ve skupini. Kromi souboru s v�sledky bude archiv obsahovat
tak� adres�r SRC/, do kter�ho ulozite soubory se zdrojov�mi k�dy
implementovan�ch syst�mu. D�le bude archiv obsahovat soubor dokumentace.pdf,
kter� bude v cesk�m, slovensk�m nebo anglick�m jazyce popisovat Vase reseni
a umozni reprodukci Vasi pr�ce. Duraz venujte tomu, jak jste syst�my behem
jejich v�voje vyhodnocovali, a kter� techniky ci rozhodnuti se pozitivni
projevily na �spesnosti syst�mu. Tento dokument bude tak� popisovat jak
ziskat Vase v�sledky pomoci prilozen�ho k�du. Bude tedy uvedeno jak Vase
zdrojov� k�dy zkompilovat, jak vase syst�my spustit, kde hledat
v�sledn� soubory, jak� pripadn� externi n�stroje je nutn� instalovat a
jak je presni pouzit, atd. Ocek�van� rozsah tohoto dokumentu jsou
3 strany A4. Do ZIP archivu prosim neprikl�dejte evaluacni data!

## WHERE TO PLAGIAT
Inspiraci pro vase syst�my muzete hledat v archivu demonstracnich prikladu
pro predmit SUR:

https://www.fit.vutbr.cz/study/courses/SUR/public/prednasky/demos/

Zvl�st se podivejte na priklad detekce pohlavi z reci: demo_genderID.py
Uzitecn� v�m mohou b�t funkce pro nacitani PNG souboru (png2fea) a extrakci
MFCC priznaku z WAV souboru (wav16khz2mfcc).

Hodnoceni:
- vse je odevzd�no a nejak�m zpusobem pracuje:
  - ctou se soubory,
  - produkuje se sk�re
  - jsou spr�vni implementov�ny a natr�nov�ny nijak� "rozumn�" detektory
    pro obr�zky a pro nahr�vky a/nebo kombinaci obou modalit (detektory
    nemusi pracovat se 100% �spisnosti, jsou to re�ln� data!)
  - jsou odevzd�ny vsechny pozadovan� soubory v pozadovan�ch form�tech.
  - v dokumentaci vysvitlite, co, jak a proc jste dilali a co by se jesti dalo zlepsit.
  ... pln� pocet 25 bodu.

- neco z v�se uveden�ho neni splnino ? ... m�ne bodu.

Posledni modifikace: 3. dubna 2022, Luk�s Burget
   
