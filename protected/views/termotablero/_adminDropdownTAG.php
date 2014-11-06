<?php

            // coloca el elemento vacío en el arreglo de opciones
            $datos=array('0' => 'Seleccione un equipo para filtrar el resultado');
            // consulta todos los modelos que tengan el area
            $datas = Tableros::model()->findAllbyAttributes(array('Area'=>$miTAG));
            $value1 = "";
            // para cada uno de los encontrados
            foreach ($datas as $modelo)
            {
                    if (isset($modelo->TAG)&&isset($modelo->Tablero)){
                        // arma el resto del arreglo de opciones con cada modelo encontrado
                        $datos[$modelo->TAG]=$modelo->TAG." - ".$modelo->Tablero;
                    }
            }
            // imprime un dropdown con las opciones en $datos
            echo CHtml::dropDownList(
                'TAG', "", $datos, 
                array(
                    'ajax' => array(
                        'type' => 'GET', //request type
                        'data' => array('TAG' => 'js:document.getElementById("TAG").value'),
                        'url' => CController::createUrl('/termotablero/dynamicTAG'), //url to call.
                        'update' => '#divGridTAG', //selector to update
                    ),
                    'onchange'=>'updateGrid',
                    'style' => 'width:100%;',
                    'class'=>'select',
                    'empty'=>'Seleccione un tablero',
                )
            );

?>
<!--TODO:falta arreglar numero de paginas para que no se vean supe rpuestas --->