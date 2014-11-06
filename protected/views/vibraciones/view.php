<?php
$this->layout = "responsiveLayout";
$this->breadcrumbs = array(
    'Vibraciones' => array('index'),
    $model->id,
);

// busca el modelo del equipo en Motores
$modelTMP = Motores::model()->findByAttributes(array('TAG' => $model->TAG));

$suffixID = "";
if (isset($modelTMP)) {
    $suffixID = $modelTMP->id;
}
$this->menu = array(
    array('label' => Yii::t('app', 'Instrucciones'), 'url' => array('/Archivos/displayArchivo?id=25')),
    array('label' => 'Lista de Mediciones', 'url' => array('index')),
    array('label' => 'Nueva Medición', 'url' => array('/vibraciones/create?id=' . $model->TAG)),
    array('label' => 'Gestionar Mediciones', 'url' => array('/vibraciones/admin?id=' . $suffixID))
);
//TODO: provisional: para uso de roles de admin, ingeniero y usuario.
$esAdmin = 0;
$esIngeniero = 0;
if (!Yii::app()->user->isGuest) {
    $modeloU = Usuarios::model()->findBySql('select * from usuarios where Username="' . Yii::app()->user->name . '"');
}
if (isset($modeloU)) {
    $esAdmin = $modeloU->Es_administrador;
    $esIngeniero = $modeloU->Es_analista;
    if ($esAdmin)
        $esIngeniero = 1;
}
if ($esIngeniero)
    array_push($this->menu, array('label' => 'Actualizar Medición', 'url' => array('update', 'id' => $model->id)));
if ($esAdmin)
    array_push($this->menu, array('label' => 'Borrar Medición', 'url' => '#', 'linkOptions' => array('submit' => array('delete', 'id' => $model->id), 'confirm' => 'EstÃ¡ seguro de borrar esto?')));
?>

<?php
$modelM = Motores::model()->findByAttributes(array('TAG' => $model->TAG));

$potencia = $modelM->kW;
$vibracionMotor = $model->VibLL;
$vibracionBomba = $model->VibLA;
$imagenVMotor = "/images/rojo.gif";
$imagenVBomba = "/images/rojo.gif";
$avisoVisible = false;
if ($potencia <= 15) {
    // Bombillo de Vib motor
    if (($vibracionMotor >= 0 ) && ( $vibracionMotor < 2.8)) {
        $imagenVMotor = "/images/verde.gif";
    } elseif (($vibracionMotor >= 2.8 ) && ( $vibracionMotor < 4.5)) {
        $imagenVMotor = "/images/amarillo.gif";
        $avisoVisible = true;
    } else {
        $imagenVMotor = "/images/rojo.gif";
        $avisoVisible = true;
    }
    // Bombillo de Vib Bomba
    if (($vibracionBomba >= 0 ) && ( $vibracionBomba < 2.8)) {
        $imagenVBomba = "/images/verde.gif";
    } elseif (($vibracionBomba >= 2.8 ) && ( $vibracionBomba < 4.5)) {
        $imagenVBomba = "/images/amarillo.gif";
        $avisoVisible = true;
    } else {
        $imagenVBomba = "/images/rojo.gif";
        $avisoVisible = true;
    }
} elseif (($potencia > 15 ) && ( $potencia < 75)) {
    if (($vibracionMotor >= 0 ) && ( $vibracionMotor < 2.8)) {
        $imagenVMotor = "/images/verde.gif";
    } elseif (($vibracionMotor >= 2.8 ) && ( $vibracionMotor < 7.1)) {
        $imagenVMotor = "/images/amarillo.gif";
        $avisoVisible = true;
    } else {
        $imagenVMotor = "/images/rojo.gif";
        $avisoVisible = true;
    }

    // Define grafico Vib Bomba
    if (($vibracionBomba >= 0 ) && ( $vibracionBomba < 2.8)) {
        $imagenVBomba = "/images/verde.gif";
    } elseif (($vibracionBomba >= 2.8 ) && ( $vibracionBomba < 7.1)) {
        $imagenVBomba = "/images/amarillo.gif";
        $avisoVisible = true;
    } else {
        $imagenVBomba = "/images/rojo.gif";
        $avisoVisible = true;
    }
} else {
    if (($vibracionMotor >= 0 ) && ( $vibracionMotor < 4.5)) {
        $imagenVMotor = "/images/verde.gif";
    } elseif (($vibracionMotor >= 4.5 ) && ( $vibracionMotor < 11)) {
        $imagenVMotor = "/images/amarillo.gif";
        $avisoVisible = true;
    } else {
        $imagenVMotor = "/images/rojo.gif";
        $avisoVisible = true;
    }

    // Define grafico Vib Bomba
    if (($vibracionBomba >= 0 ) && ( $vibracionBomba < 4.5)) {
        $imagenVBomba = "/images/verde.gif";
    } elseif (($vibracionBomba >= 4.5 ) && ( $vibracionBomba < 11)) {
        $imagenVBomba = "/images/amarillo.gif";
        $avisoVisible = true;
    } else {
        $imagenVBomba = "/images/rojo.gif";
        $avisoVisible = true;
    }
}


$nombre = "";
$modelTMP = Vibraciones::model()->findByAttributes(array('TAG' => $model->TAG));
if (isset($modelTMP)) {

    if (isset($modelTMP->TAG))
        $nombre = $modelTMP->TAG;
    if (isset($modelTMP->TAG))
        $nombre = $nombre . ' (' . $modelTMP->Fecha . ')';
}

?>
<!-- Título de la página -->
<?php $this->setPageTitle ('Detalles de medición:'.$nombre); ?>

<div id="detailV" class="forms50c forms50cl"><span>
    <?php
    $this->widget('zii.widgets.CDetailView', array(
        'data' => $model,
        'cssFile' => '/themes/detailview/styles.css',
        'attributes' => array(
            // 'id',
            //  'Toma',
            //'TAG',
            array(// related city displayed as a link
                'label' => 'Motor',
                'type' => 'raw',
                'value' => ((isset($modelM->TAG)) ? $modelM->TAG : "") . " - " . ((isset($modelM->Motor)) ? $modelM->Motor : ""),
            ),
            array(// related city displayed as a link
                'label' => 'Equipo',
                'type' => 'raw',
                'value' => (isset($modelM->Equipo)) ? $modelM->Equipo : "",
            ),
            array(// related city displayed as a link
                'label' => 'Proceso',
                'type' => 'raw',
                'value' => (isset($modelM->Area)) ? $modelM->Area : "",
            ),
            array(// related city displayed as a link
                'label' => 'Area',
                'type' => 'raw',
                'value' => (isset($modelM->Proceso)) ? $modelM->Proceso : "",
            ),
            array(// related city displayed as a link
                'label' => 'Orden de Trabajo',
                'type' => 'raw',
                'value' => "<b>" . ((isset($model->OT)) ? $model->OT : "") . "</b>",
            ),
            array(// related city displayed as a link
                'label' => 'Potencia',
                'type' => 'raw',
                'value' => "" . ((isset($modelM->kW)) ? $modelM->kW : "") . " kW",
            ),
        ),
    ));
    ?>
        </span>
</div>

<style>
    .bordesAmarillos{
        padding:0px;margin:0px;border-color:#FCF4A1;border-width: 1px;border-style:solid;
    }
</style>

<div class="forms50c forms50cr">
    <?php
    // Crea una tabla con gridview para poder usar su mismo tema.
        $tablaDatos = array(
            array('id' => 1, 'Motor' => ((isset($model->VibLL)) ? $model->VibLL : ""), 'Reductor_Bomba' => ((isset($model->VibLA)) ? $model->VibLA : ""), 'Mayor_Valor' => ((isset($model->Temperatura)) ? $model->Temperatura : "")), //Fila 1
            array('id' => 2, 'Motor' => CHTML::image($imagenVMotor), 'Reductor_Bomba' => CHTML::image($imagenVBomba), 'Mayor_Valor' => ''), //Fila 2
        );
        $dataProvider = new CArrayDataProvider($tablaDatos, array('id' => 'userTD'));
        $this->widget('zii.widgets.grid.CGridView', array('id' => 'tablaD','dataProvider' => $dataProvider, 'cssFile' => '/themes/gridview/styles.css',     'template'=> '{items}{pager}{summary}',     'summaryText'=>'Resultados del {start} al {end} de {count} encontrados','template' => '{items}',
            'columns' => array(
                'Motor:raw:Motor',
                'Reductor_Bomba:raw:Motor o Bomba',
                'Mayor_Valor:raw:Mayor Valor',
            ),
            'htmlOptions'=>array('style'=>'height:144px;'),
            ));
    ?>
</div>





<?php
// guarda en $TAG_in el TAG del modelo actual
// para admin $model=Vibraciones::model()->findByAttributes(array('TAG'=>$TAG));
$TAG_in = $model->TAG;
if ($TAG_in == "") {
    echo "No hay datos para graficar.";
} else {
    ?>

    <?php

// arreglos para gaficar
    $arrSalidaTemperatura = array();
// obtiene una array de modelos que tienen el tag= al del modelo actual
    $data = Vibraciones::model()->findAllBySQL('SELECT * FROM VIBRACIONES WHERE TAG="'.$model->TAG.'" ORDER BY FECHA ASC');
// para cada uno de los modelos calcula el IP de cada fase y lo almacena en formato de stockhigh
    foreach ($data as $modeloTemp) {
        // hace push en arrSalidaA,B,C de un array(strtotime($foreignobj->Fecha)*1000,$foreignobj->A10/$foreignobj->A1))
        array_push($arrSalidaTemperatura, array(strtotime($modeloTemp->Fecha) * 1000, 0 + $modeloTemp->Temperatura));
    }
//print_r($arrSalidaTemperatura);
// hace el gráfico de IP
    ?>
    <div style="width:100%; height:400px;display: inline-block;" class="graphs">
        <?php
        $this->Widget('ext.highstock.HighstockWidget', array(
            'id' => 'grafico2',
            'options' => array(
                'colors' => array('#3A48F9'),
                'chart' => array(
                    'selectionMarkerFill' => '#F6EAA5',
                    'backgroundColor' => '#FCFAE8',
                    'borderColor' => '#AC8B3A',
                    'borderWidth' => 1,
                    'borderRadius' => 8,
                    //'width'=>400,
                    // 'height'=>300,
                    'plotBackgroundColor' => '#FFFFFF',
                ),
                'credits' => array('enabled' => false),
                'legend' => array('enabled' => false),
                'theme' => 'grid',
                'rangeSelector' => array('selected' => 2,
                    'buttons' => array(
                        array('type' => 'month',
                            'count' => 1,
                            'text' => '1m'
                        ),
                        array('type' => 'all',
                            'text' => 'All'
                        )
                    )
                ),
                'colors' => array('#6caddf', '#8bc63e', '#fec422'),
                'credits' => array('enabled' => false),
                'legend' => array('enabled' => false),
                'theme' => 'grid',
                'title' => array('text' => 'Tendencia Temperatura más Alta en el Conjunto Motriz'),
                'xAxis' => array('maxZoom' => 14 * 24 * 360000),
                'yAxis' => array('title' => array('text' => 'Temperatura')),
                'series' => array(
                    array('name' => 'Tendencia Temperatura más Alta en el Conjunto Motriz', 'data' => $arrSalidaTemperatura)
                ),
                'plotOptions' => array(
                    'series' => array(
                        'marker' => array(
                            'enabled' => true,
                        ),
                    ),
                ),
                'htmlOptions' => array('z-index' => '-1'),
                ))
        );
        ?>
    </div>
    <div style="width:49%; display: inline-block;float:left"  class="graphs">

    <?php
// guarda en $TAG_in el TAG del modelo actual


// arreglos para gaficar
    $arrSalidaVibLL = array();
    
// para cada uno de los modelos calcula el IP de cada fase y lo almacena en formato de stockhigh
    foreach ($data as $modeloTemp) {
        // hace push en arrSalidaA,B,C de un array(strtotime($foreignobj->Fecha)*1000,$foreignobj->A10/$foreignobj->A1))
        array_push($arrSalidaVibLL, array(strtotime($modeloTemp->Fecha) * 1000, 0 + $modeloTemp->VibLL));
    }
//print_r($arrSalidaTemperatura);
// hace el gráfico de IP
    $this->Widget('ext.highstock.HighstockWidget', array(
        'id' => 'grafico3',
        'options' => array(
            'colors' => array('#3A48F9'),
            'credits' => array('enabled' => false),
            'legend' => array('enabled' => false),
            'theme' => 'grid',
            'rangeSelector' => array(
                'selected' => 2,
                'buttons' => array(
                    array('type' => 'month',
                        'count' => 1,
                        'text' => '1m'
                    ),
                    array('type' => 'all',
                        'text' => 'All'
                    )
            )),
            'colors' => array('#8bc63e', '#fec422'),
            'chart' => array(
                'backgroundColor' => '#FCFAE8',
                'borderColor' => '#AC8B3A',
                'borderWidth' => 1,
                'borderRadius' => 8,
                //  'width'=>400,
                // 'height'=>300,
                'plotBackgroundColor' => array(
                    'linearGradient' => array(0, 300, 0, 350),
                    'stops' => array(
                        array(0, 'rgb(255, 255, 255)'),
                        array(1, 'rgb(0, 0, 0)')
                    )
                ),
                'style' => array(
                    'color' => '#ffffff',
                    'fontSize' => '8px',
                )
            ),
            'title' => array('text' => 'Vibraciones en el Motor'),
            'xAxis' => array('maxZoom' => 14 * 24 * 360000),
            'yAxis' => array('title' => array('text' => 'Vibraciones')),
            'series' => array(
                array('name' => 'Tendecia Vibraciones Motor ', 'data' => $arrSalidaVibLL)
            ),
            'plotOptions' => array(
                'series' => array(
                    'marker' => array(
                        'enabled' => true,
                    ),
                ),
            ),
            ))
    );
    ?>

    </div>
    <div style="width:49%;display: inline-block;float:right;" class="graphs">

    <?php
// guarda en $TAG_in el TAG del modelo actual


// arreglos para gaficar
    $arrSalidaVibLA = array();
// para cada uno de los modelos calcula el IP de cada fase y lo almacena en formato de stockhigh
    foreach ($data as $modeloTemp) {
        // hace push en arrSalidaA,B,C de un array(strtotime($foreignobj->Fecha)*1000,$foreignobj->A10/$foreignobj->A1))
        array_push($arrSalidaVibLA, array(strtotime($modeloTemp->Fecha) * 1000, 0 + $modeloTemp->VibLA));
    }
//print_r($arrSalidaTemperatura);
// hace el gráfico de IP
    $this->Widget('ext.highstock.HighstockWidget', array(
        'id' => 'grafico8',
        'options' => array(
            'colors' => array('#fec422'),
            'credits' => array('enabled' => false),
            'legend' => array('enabled' => false),
            'theme' => 'grid',
            'rangeSelector' => array(
                'selected' => 1,
                'buttons' => array(
                    array('type' => 'month',
                        'count' => 1,
                        'text' => '1m'
                    ),
                    array('type' => 'all',
                        'text' => 'All'
                    ),
            )),
            'chart' => array(
                'selectionMarkerFill' => '#F6EAA5',
                'backgroundColor' => '#FCFAE8',
                'borderColor' => '#AC8B3A',
                'borderWidth' => 1,
                'borderRadius' => 8,
                //  'width'=>400,
                // 'height'=>300,
                'plotBackgroundColor' => '#FFFFFF',
                'style' => array(
                    'color' => '#ffffff',
                    'fontSize' => '8px',
                ),
            ),
            'title' => array('text' => 'Vibraciones Lado Bomba o Reductor'),
            'xAxis' => array('maxZoom' => 14 * 24 * 360000),
            'yAxis' => array('title' => array('text' => 'Vibraciones')),
            'series' => array(
                array('name' => 'Tendencia Vibraciones Lado Bomba o Reductor ', 'data' => $arrSalidaVibLA)
            ),
            'plotOptions' => array(
                'series' => array(
                    'marker' => array(
                        'enabled' => true,
                    ),
                ),
            ),
            ))
    );
    ?>

    </div>

<?php } ?>