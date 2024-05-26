def modificar_tipo_si_white_blood_cell(archivo_entrada, archivo_salida):
    with open(archivo_entrada, 'r') as archivo_lectura, open(archivo_salida, 'w') as archivo_escritura:
        for linea in archivo_lectura:
            partes = linea.strip().split(',')
            # Verificamos si la tercera columna es 'white_blood_cell'
            if partes[2] == "white_blood_cell":
                # Cambiamos el tipo en la segunda columna a 'uninfected'
                partes[1] = "uninfected"
                # Reconstruimos la línea con el tipo modificado
                linea_modificada = ",".join(partes)
                archivo_escritura.write(linea_modificada + "\n")
            else:
                # Si no es 'white_blood_cell', escribimos la línea tal cual
                archivo_escritura.write(linea)

# Suponiendo que el archivo de entrada se llama "datos.txt"
# y queremos guardar los cambios en un archivo nuevo llamado "datos_modificados.txt"
modificar_tipo_si_white_blood_cell("dataset12/test/test.txt", "dataset12/test/test_modificado.txt")
