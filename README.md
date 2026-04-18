<img width="4757" height="2187" alt="ek-logo-szoveggel" src="https://github.com/user-attachments/assets/b013ac70-753a-4b6c-a1cc-0d6228bdf9b7" />

## Képtömörítés-Quadtree-algoritmussal

Ez a projekt az **Európa Kollégium Informatika-műhelyének** keretén belül készült.

## A projekt célja

A projekt célja egy **Quadtree-alapú képtömörítési algoritmus** implementálása Python nyelven, amely rekurzív módon bontja fel a képet homogén régiókra, majd ezek alapján tömörített reprezentációt készít.

A program támogatja:
- színes (RGB) képek feldolgozását
- tömörített kép visszaállítását
- fázisképek generálását
- Quadtree struktúra vizualizálását

---

## Működés röviden

A Quadtree algoritmus a képet addig osztja 4 részre, amíg a blokkok nem lesznek elég homogének.

### INPUT
- Egy bemeneti kép (`.jpg`, `.png`, stb.)

### OUTPUT
- Tömörített kép
- Fázisképek különböző mélységekhez
- Quadtree fa vizualizáció
- Statisztikai mérőszámok

---

## Az algoritmus lépései

1. A képet RGB formátumban betöltjük.
2. Egy adott régióra kiszámítjuk:
   - átlagos színt
   - varianciát
3. Ha a variancia ≤ threshold → a blokk homogén → nem bontjuk tovább
4. Ha nem homogén → 4 részre bontjuk (rekurzívan)
5. A levelekből visszaépítjük a képet

---

## Mérőszámok

A program automatikusan kiszámolja:

- **MSE (Mean Squared Error)** – hiba az eredeti és tömörített kép között
- **PSNR (Peak Signal-to-Noise Ratio)** – képminőség
- **Futási idő**
- **Levélcsomópontok száma**
- **Fa mélysége**
- **Becsült tömörítési arány**

---
Kiss Csenge<br>
Újvidéki Egyetem, Természettudományi-Matematikai Kar, hallgató

Műhelyvezető:<br>
Dr. Pintér Róbert

---

