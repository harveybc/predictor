<?php $this->pageTitle = Yii::app()->name; ?>


<?php
$this->pageTitle = Yii::app()->name . ' - Administrar';
$this->breadcrumbs = array(
    'Administrar',
);
?>
<h><b>Por favor seleccione uno de los siguientes items para realizar operaciones (crear, borrar, editar, buscar o listar):</b></h>



<?php

/*
$dataTree=array(
		array(
			'text'=>'Grampa', //must using 'text' key to show the text
			'children'=>array(//using 'children' key to indicate there are children
				array(
					'text'=>'Father',
					'children'=>array(
						array('text'=>'<hr />maaIe<hr />'),
						array('text'=>'big sis<hr />'),
						array('text'=>'little brother<hr />'),
					)
				),
				array(
					'text'=>'Uncle',
					'children'=>array(
						array('text'=>'Ben'),
						array('text'=>'Sally'),
					)
				),
				array(
					'text'=>'Aunt',
				)
			)
		)
	);
 */
// coloca el datatree vacío.        
$dataTree=array();
// encuentra todas las áreas (columna proceso)
$areas=Estructura::model()->findAllBySql("SELECT DISTINCT Proceso FROM estructura ORDER BY Proceso");
// para cada área adiciona una entrada y sus acciones, 
$na=0;
foreach($areas as $area){
    array_push($dataTree, array(
        // faltan opciones
        'text'=>$area->Proceso,
        'children'=>array(),
    ));
    $procesos=Estructura::model()->findAllBySql('SELECT DISTINCT Area FROM estructura WHERE Proceso="'.$area->Proceso.'" ORDER BY Area');
    // para cada proceso
    $np=0;
    foreach ($procesos as $proceso){
        array_push($dataTree[$na]['children'],array(
            'text'=>$proceso->Area,
            'children'=>array(
                array('text'=>'Equipos','children'=>array()),
                array('text'=>'Tableros','children'=>array()),
            ),
        ));
        $tableros=  Tableros::model()->findAllBySql('SELECT Tablero FROM tableros WHERE Area="'.$proceso->Area.'" ORDER BY Tablero');
        // para cada tablero
        foreach ($tableros as $tablero){
            array_push($dataTree[$na]['children'][$np]['children'][1]['children'],array(
                'text'=>$tablero->Tablero,
                'children'=>array(),
            ));
        }
        $equipos=Estructura::model()->findAllBySql('SELECT Equipo FROM estructura WHERE Area="'.$proceso->Area.'" ORDER BY Equipo');
        // para cada proceso
        $ne=0;
        foreach ($equipos as $equipo){
            array_push($dataTree[$na]['children'][$np]['children'][0]['children'],array(
                'text'=>$equipo->Equipo,
                'children'=>array(),
            ));
            $motores=Motores::model()->findAllBySql('SELECT Motor FROM motores WHERE Equipo="'.$equipo->Equipo.'" ORDER BY Motor');
            foreach ($motores as $motor){
                array_push($dataTree[$na]['children'][$np]['children'][0]['children'][$ne]['children'],array(
                    'text'=>$motor->Motor.' - '.$motor->TAG,
                    'children'=>array(),
                ));
            }
            $ne++;
        }
        $np++;
    }
    $na++;
}
//print_r($dataTree);
/*
    // agrega el arreglo de hijos
    $procesos=Estructura::model()->findAllBySql('SELECT DISCTINT Area FROM estructura WHERE Proceso="'.$area->Proceso.'" ORDER BY Area');
    // para cada área adiciona una entrada y sus acciones,
    
    foreach($procesos as $proceso){
        array_push($dataTree, array(
            $dataTree['']
        ));
    
        // para los equipos del proceso, adiciona una entrada y sus acciones,
            // para los motores del equipo, adiciona una entrada y sus acciones
    }

    
$this->widget('CTreeView', array(
        'htmlOptions' => array(
            'class' => 'treeview-famfamfam'
        ),
        'id' => 'treeviewId',
        
        'data' =>$dataTree,
            
            
        
        'unique' => true,
        //'persist' => 'cookie',
        //'prerendered' => false,}
        'collapsed' => true,
        //'control' => '#selector',
        //'url' => array('ajaxFillTree')
));
*/
$this->widget("application.extensions.widgets.jstree.JSTree", array(
    "plugins"=>array(
        "html_data"=>
            array("data"=>$model->getGeozoneTreeUL($model->getGeozone_ids())),
        "checkbox"=>
            array(
                        "real_checkboxes"=>true,
                        "two_state"=>true,
                        "real_checkboxes_names"=>'js:function (n) {
                           var val = n[0].id.replace(/geozone_classification_tree_/, "");
                           return [("check_geozone_" + (n[0].id)), val];
                         }'),
            "themes"=>array( "theme" => "default" ),
            "sort",
            "ui"
    ),
    "model"=>$model->getTitleGeozoneModel(),
    "attribute"=>"geozones_id"
));
?>


