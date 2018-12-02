# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 17:47:34 2018

@author: alelu
"""

import glob
import glymur
import cv2
import numpy as np
import operator
import os
import csv

from skimage.feature import peak_local_max
from skimage import morphology
from skimage.measure import regionprops
from scipy import ndimage

class CellDensity:
    
    """
    Metodo che legge tutti i tile da un immagine in formato jp2 , e li ordina in base 
    al numero di pixel con valore 1dell'immagine calcolata tramite sogliatura.
    Si può decidere di salvare tutti i tile durante la fase di elaborazione
   , oppure di salvare solo quelli restituiti come output.
    Utilizza i metodi di classe
        save_csv_file
        count_white_pixel
        order
        write_ordered_tiles
    """
    def order_tile_by_thresh_density(self, jp2_file, output_folder, n = 200, save_tiles = False):
        jp2 = glymur.Jp2k(jp2_file)
        i = 0
        images_white_pixel = []
        for i in range(0, self.compute_number_of_tiles(jp2_file)):
            if save_tiles == True:
                self.__write_tile(jp2.read(tile = i), i, output_folder)
            images_white_pixel.append((i, self.__count_white_pixel(jp2.read(tile = i), True)))
            i += 1
        self.__save_csv_file(images_white_pixel, output_folder, "not-ordered")
        self.__order_by_white_pixel(images_white_pixel)
        self.__save_csv_file(images_white_pixel, output_folder, "ordered")
        self.__write_ordered_tiles(jp2, images_white_pixel, output_folder, n)
        
    """
    Metodo che salva un tile all'interno di una cartella di output selezionata.
    Fa uso della funzione imwrite della libreria opencv , e delle funzioni os 
    per verificare che le cartelle selezionate esistano prima del salvataggio.
    """
    def __write_tile(self, tile, i, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if not os.path.exists("{}/tiles".format(output_folder)):
            os.makedirs("{}/tiles".format(output_folder))
        cv2.imwrite("{}/tiles/tile-{}".format(output_folder, i), tile)
    
    """
    Metodo che dopo l'ordinamento in base al numero di pixel con valore 1 
    dei tiles analizzati, salva all'interno di una cartella di output selezionata ,
    un numero n , scelto dall'utente, di tile ordinati.
    Fa uso della funzione imwrite della libreria opencv , e delle funzioni os 
    per verificare che le cartelle selezionate esistano prima del salvataggio.
    """    
    def __write_ordered_tiles(self, jp2, images_white_pixel, output_folder, n):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if not os.path.exists("{}/output-tiles".format(output_folder)):
            os.makedirs("{}/output-tiles".format(output_folder))
        for i in range(0, n):
            index = images_white_pixel[i][0]
            cv2.imwrite("{}/output-tiles/output-tile-{}.png".format(output_folder, i), cv2.cvtColor(jp2.read(tile = index),cv2.COLOR_BGR2RGB))
        pass
    
    """
    Metodo che legge dall'immagine JPEG 2000 i valori di altezza e larghezza,
    e legge gli stessi valori di un tile presente all'interno dell'immagine.
    Tramite la seguente formuula :
        (pixel in larghezza immagine * pixel in altezza immagine)/(pixel in larghezza tessera * pixel in altezza tessera) 
    calcola il numero di tile presenti all'interno dell'immagine OPEN JPEG
    """
    def compute_number_of_tiles(self, jp2_file):
        jp2 = glymur.Jp2k(jp2_file)
        jp2_height , jp2_width , jp2_channels = jp2.shape
        tile_height, tile_width, tile_channels = jp2.read(tile = 0).shape
        size =(int) ((jp2_height * jp2_width) / (tile_height * tile_width)) - 1
        return size
    
    """
    Metodo che legge tutte le immagini presenti in una cartella , e le ordina 
    in base al numero di pixel con valore 1 dell'immagine calcolata tramite sogliatura.
    Fa uso dei metodi di classe:
        count_white_pixel
        save_csv_file
        order
        write_ordered_image
    """
    def order_img_by_thresh_density(self, input_folder, output_folder):
        i = 0
        images = self.__read_images(input_folder)
        images_white_pixel  = []
        for image in images:
            images_white_pixel.append((i, self.__count_white_pixel(image)))
            i += 1
        self.__save_csv_file(images_white_pixel, output_folder, 'not-ordered-by-thresh')
        self.__order_by_white_pixel(images_white_pixel)
        self.__save_csv_file(images_white_pixel, output_folder, 'ordered-by-thresh')
        self.__write_ordered_image(images, images_white_pixel, output_folder, "ordered-by-thresh")
        
    """
    Metodo che conta i pixel con valore 1 in una immagine segmentata tramite sogliatura.
    Utilizza varie funzioni della libreria opencv:
        cv2.cvtColor per trasformare 
        cv2.GaussianBlur per ridurre i rumori nell'immagine
        cv2.threhold per applicare la segmentazione tramite sogliatura all'immagine utilizzando il metodo di OTSU
        cv2.erode effettua l'erosione dell'immagine segmentata tramite sogliatura
        cv2.dilate effettua la dilatazione dell'immagine segmentata tramite sogliatura
        cv2.countNonZero per contare il numero di pixel diversi da 0 nell'immagine segmentata
    Restituisce il numero di pixel bianchi ritrovati.
    """
    def __count_white_pixel(self, image, tile = False):
        if tile == True:
            gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        cv2.erode(thresh, None, iterations=13)
        cv2.dilate(thresh, None, iterations=13)
        white_pixel = cv2.countNonZero(thresh)
        return white_pixel #images_white_pixel.append((i,white_pixel))
    
    """
    Metodo che legge tutte le immagini presenti in una cartella , e le ordina in base 
    al numero di cellule estratte dall'immagine dopo vari procedimenti. 
    Permette di salvare le cellule estratte se voluto dall'utente.
    Fa uso dei metodi di classe:
        count_cells
        count_white_pixel
        save_csv_file
        order
        write_ordered_image
    """    
    def order_img_by_extracted_cell(self, input_folder, output_folder, save_file = False):
        i = 0
        images = self.__read_images(input_folder)
        images_region_count  = []
        for image in images:
            filtered_labels = self.__count_cells(image, save_file)
            images_region_count.append((i, len(regionprops(filtered_labels))))
            i += 1
        self.__save_csv_file(images_region_count, output_folder, 'not-ordered-by-extracted-cell')
        self.__order_by_white_pixel(images_region_count)
        self.__save_csv_file(images_region_count, output_folder, 'ordered-by-exctracted-cell')
        self.__write_ordered_image(images, images_region_count, output_folder, "ordered-by-extracted-cell")
        if save_file == True:
            for item in images_region_count:
                if not os.path.exists("{}/ordered-by-extracted-cell/extracted-cells/img-{}".format(output_folder, item[0])):
                    os.makedirs("{}/ordered-by-extracted-cell/extracted-cells/img-{}".format(output_folder, item[0]))
                image = cv2.imread("{}/ordered-by-extracted-cell/img-{}.png".format(output_folder, item[0]))
                self.__count_cells(image, save_file, "{}/ordered-by-extracted-cell/extracted-cells/img-{}".format(output_folder, item[0]))
        
    """
    Legge tutte le immagini presenti in una cartella e le salva all'interno di una lista.
    Utilizza il metodo glob della libreria glob per selezionare tutti i file con estensione 
    .png presenti all'interno della cartella di input selezionata.
    Utilizza il metodo imread della libreria opencv per leggere le immagini.
    """
    def __read_images(self, input_folder):
        images = []
        for image in glob.glob("{}/*.png".format(input_folder)):
            current_image = cv2.imread(image)
            images.append(current_image)
        return images
        
    """
    Ordina la lista contenente le immagini o i tiles 
    """        
    def __order_by_white_pixel(self, images_white_pixel):
        images_white_pixel.sort(key=operator.itemgetter(1))
        images_white_pixel.reverse()
        
    """
    Salva le immagini ordinate in ordine decrescente all'interno di una cartella 
    selezionata dall'utente.
    Utilizza il metodo imwrite della libreria opencv per salvare le immagini.
    """
    def __write_ordered_image(self, images, images_white_pixel, output_folder, folder_name):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if not os.path.exists("{}/{}".format(output_folder, folder_name)):
            os.makedirs("{}/{}".format(output_folder, folder_name))
        for i in range(0, len(images_white_pixel)):
            index = images_white_pixel[i][0]
            cv2.imwrite("{}/{}/img-{}.png".format(output_folder, folder_name, i), images[index])
            
    """
    Applica la segmentazione tramite sogliatura all'immagine , dopo di che effettua due trasformazioni morfologiche di erosione
    e dilatazione per definire meglio i contorni cellulari. In seguito applica l'algoritmo di Canny per definire i contorni
    delle cellule prsente dell'immagine segmentata. Con i contorni ottenuti crea una maschera che verrà utilizzata durante
    l'applicazione dell'algoritmo di watershed in modo da farlo lavorare solo sulle aree interessate
    Utilizza varie funzioni della libreria opencv:
        cv2.cvtColor per cambiare il modello di colore da BGR a scala di grigi
        cv2.GaussianBlur per ridurre i rumori nell'immagine
        cv2.threhold per applicare la segmentazione tramite sogliatura all'immagine utilizzando il metodo di OTSU
        cv2.erode effettua l'erosione dell'immagine segmentata tramite sogliatura
        cv2.dilate effettua la dilatazione dell'immagine segmentata tramite sogliatura
        cv2.Canny per applicare l'algoritmo di Canny all'immagine
    Restituisce la maschera 
    """        
    def __cells_mask(self, image):
        meanshift = cv2.pyrMeanShiftFiltering(image, 21 , 51)
        gray = cv2.cvtColor(meanshift, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        erode = cv2.erode(thresh, None, iterations=3)
        dilate = cv2.dilate(thresh, None, iterations=3)
        ret, dilate_thresh = cv2.threshold(dilate, 1, 128, 1)
        marker = cv2.add(erode, dilate_thresh)
        canny = cv2.Canny(marker, 110, 150)
        new, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        marker32 = np.int32(marker)
        m = cv2.convertScaleAbs(marker32)
        ret, thresh = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thresh_inv = cv2.bitwise_not(thresh)
        return cv2.bitwise_and(image, image, mask=thresh_inv)
    
    """
    Salva le cellule ritrovate all'interno dell'immagine dopo che è stato applicato 
    l'algoritmo di watershed.
    Utilizza il metodo imwrite della libreria opencv per salvare le immagini.
    """
    def __split_cell_labels(self, gray, original_image, filtered_labels, output_folder):
        i = 0
        for region in regionprops(filtered_labels):
            minr, minc, maxr, maxc = region.bbox

            # Transform the region to crop from rectangular to square
            x_side = maxc - minc
            y_side = maxr - minr
            if x_side > y_side:
                maxr = x_side + minr
            else:
                maxc = y_side + minc

            if (minc > 20) & (minr > 20):
                minc = minc - 20
                minr = minr - 20
                
            if i != 0:
                cropped = np.array(original_image[minr:maxr + 20, minc:maxc + 20])
                cv2.imwrite("{}/{}.png".format(output_folder, i - 1), cropped)
            i += 1
            
    """
    Esclude le aree ritrovate dall'algoritmo di watershed che non 
    hanno una dimensione adeguata.
    Un componente deve avere una dimensione maggiore di 2000 e minore di 62000.
    Restituisce le aree filtrate.
    """
    def __filter_labels(self, labels):
        filtered_labels = np.copy(labels)
        component_sizes = np.bincount(labels.ravel())
        too_small = component_sizes < 2000
        too_small_mask = too_small[labels]
        filtered_labels[too_small_mask] = 1
        too_big = component_sizes > 62000
        too_big_mask = too_big[labels]
        filtered_labels[too_big_mask] = 1
        return filtered_labels

    """
    Richiama il metodo watershed_and_filtering , potrebbe essere eliminato 
    nelle versioni future.
    Restituisce le zone filtrate
    """
    def __count_cells(self, image, save_file, output_folder = ""):
        filtered_labels = self.__watershed_and_filtering(image, save_file, output_folder)
        return filtered_labels
    
    """
    Metodo che applica l'algoritmo di watershed all'immagina restituita dal metodo cell_mask.
    Le aree trovate dall'algoritmo vengono poi inviate al metodo split_cell che si occupa di estrarre le zone e salvarle
    all'interno dei rispettivi file.
     Utilizza varie funzioni della libreria opencv:
        cv2.cvtColor per cambiare il modello di colore da BGR a scala di grigi 
        cv2.threhold per applicare la segmentazione tramite sogliatura all'immagine utilizzando il metodo di OTSU
        morphology.disk crea elementi a forma di disco 
        morphology.dilation effettua la dilatazione dell'immagine segmentata tramite sogliatura
        ndimage.distance_transform_edt crea una mappa delle distanze , ogni pixel contiene la distanza euclidea da ogni pixel
        sul bordo
        peak_local_max restituisce il massimo locale nella mappa delle distanze
        morphology_watershed applica l'algoritmo di watershed
        filter_labels richiama il metodo filter_labels per filtrare le zone non utili
    Restiuisce le zone filtrate
    """
    def __watershed_and_filtering(self, image, save_file, output_folder):
        cell_mask = self.__cells_mask(image)
        gray = cv2.cvtColor(cell_mask, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        selem = morphology.disk(5)
        thresh = morphology.dilation(thresh, selem)
        D = ndimage.distance_transform_edt(thresh)
        localMax = peak_local_max(D, indices=False, min_distance=35,
                                   labels=thresh)
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = morphology.watershed(-D, markers, mask=thresh)
        filtered_labels = self.__filter_labels(labels)
        if save_file == True:
            self.__split_cell_labels(gray, image, filtered_labels, output_folder)
        return filtered_labels
    
    """
    Metodo che crea un file csv contenenti i dati presenti nelle liste , 
    contenenti tuple indice-valore, calcolate durante la fase di ordinamento 
    delle immagini. Utilizza la libreria os per creare le cartelle , e la libreria csv
    per creare e popolare il file.
    """
    def __save_csv_file(self, data, output_folder, file_name):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        with open("{}/{}.csv".format(output_folder, file_name), "w", newline = "") as f:
            writer = csv.writer(f)
            for item in data:
                writer.writerow([item[0],item[1]])
              
    """
    Metodo che legge da  un file csv, 
    contenente tuple indice-valore, calcolate durante la fase di ordinamento 
    delle immagini. Utilizza la libreria os per creare le cartelle , e la libreria csv
    per leggere da file, e la libreria cv2 per salvare le immagini.
    """
    def save_tile_from_csv_file(self, jp2_file, csv_file, output_folder, n = 200):
        jp2 = glymur.Jp2k(jp2_file)
        images_white_pixel = []
        try:
          with open(csv_file, "r", newline = "") as f:
            reader = csv.reader(f, delimiter=",")
            for item in reader:
                i = int(item[0])
                white_pixel = int(item[1])
                images_white_pixel.append((i, white_pixel))
          print(images_white_pixel)
          self.__write_ordered_tiles(jp2, images_white_pixel, output_folder, n)
        except FileNotFoundError:
            print("File not founded!")
     
    """
    Metodo che crea un'anteprima dell'immagine del vetrino di dimensione 8 volte minore
    rispetto all'originale.
    Utilizza la libreria glymur per la lettura dell'immagine JPEG 2000 , e la libreria opencv
    per il salvataggio.
    """
    def save_jp2_preview(self, jp2_file, output_folder):
         jp2 = glymur.Jp2k(jp2_file)
         cv2.imwrite("{}/preview.png".format(output_folder), cv2.cvtColor(jp2[::8, ::8],cv2.COLOR_BGR2RGB))
         
        
#PREVIEW
         
jp2_file = "PERCORSO FILE JPEG 2000"
output_folder = "CARTELLA DI OUTPUT"            

cell_density = CellDensity()
cell_density.save_jp2_preview(jp2_file, output_folder)     
            
#ELABORAZIONE DI UN FILE JP2

jp2_file = "PERCORSO FILE JPEG 2000"
tile_output_folder = "CARTELLA DI OUTPUT PER I TILE"            

cell_density = CellDensity()
cell_density.order_tile_by_thresh_density(jp2_file, tile_output_folder)


#ELABORAZIONE DI UNA CARTELLA D'IMMAGINI
input_folder = "CARTELLA DA DOVE RECUPERARE LE IMMAGINI DA ANALIZZARE"
output_folder = "CARTELLA DOVE SALVARE LE IMMAGINI ANALIZZATE"

cell_density = CellDensity()
cell_density.order_img_by_thresh_density(input_folder, output_folder)
cell_density.order_img_by_extracted_cell(input_folder, output_folder, True)

#ELABORAZIONE DA FILE CSV

jp2_file = "PERCORSO FILE JPEG 2000"
csv_file = "PERCORSO FILE CSV"
output_folder = "CARTELLA DOVE SALVARE LE IMMAGINI DI OUTPUT"

cell_density = CellDensity()
cell_density.save_tile_from_csv_file(jp2_file, csv_file, output_folder)




