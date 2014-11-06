
   <?php
// guarda en $TAG_in el TAG del modelo actual
// para admin $model=Vibraciones::model()->findByAttributes(array('TAG'=>$TAG));

if ($TAG == "") {
    echo "No hay datos para graficar.";
} else {
    ?>
<style>
    .bordesAmarillos{
        padding:0px;margin:0px;border-color:#AC8B3A;border-width: 1px;border-style:solid;
    }
</style>            


<div class="forms50cb forms50cr" style="text-align:center;">

    <?php
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
$dependeT = "{view}";
if ($esIngeniero)
    $dependeT = "{view}{update}";
if ($esAdmin)
    $dependeT = "{view}{update}{delete}";

$model = new Vibraciones('search');

$vibracionMotor = $model->VibLL;
$vibracionBomba = $model->VibLA;
    $imagenVMotor = "/images/fallback2.gif";
    $imagenVBomba = "/images/fallback2.gif";
    $avisoVisible = false;
    $modeloLast = new Vibraciones;
    $modeloLast = Vibraciones::model()->findBySql("select * from vibraciones where TAG=\"" . $TAG . "\" order by Fecha Desc");
    if (isset($modeloLast)) {
        $modelM = Motores::model()->findByAttributes(array('TAG' => $modeloLast->TAG));
        $vibracionMotor = $modeloLast->VibLL;
        $vibracionBomba = $modeloLast->VibLA;
    }
    if (isset($modelM))
        $potencia = $modelM->kW;
    if ((isset($modelM)) && (isset($modeloLast))) {

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
    }
    ?>  

    <table style="width: 100%; border-color:#FCF4A1;border-width: 1px;border-style:solid;padding: 0px;margin:0px;">
        <tr>
            <td colspan="2" style="width: 80px; text-align: center" class="bordesAmarillos">
                <span style="color: #ff0033">Vibraciones (mm/s)</span></td>
            <td style="width: 80px; text-align: center" class="bordesAmarillos">
                <span style="color: #ff0033">Temperatura(ºC)</span></td>
        </tr>
        <tr>
            <td style="width: 80px; text-align: center" class="bordesAmarillos">
                <em><b>Motor</b></em></td>
            <td style="width: 120px; text-align: center" class="bordesAmarillos"">
                <em><b>Reductor o Bomba</b></em></td>
            <td style="width: 80px; text-align: center" class="bordesAmarillos">
                <em><b>Mayor Valor</b></em></td>
        </tr>
        <tr>
            <td style="width: 80px; text-align: center" class="bordesAmarillos">
<?php echo isset($modeloLast) ? (isset($model->VibLL)) ? $modeloLast->VibLL : "" : "No se encontraron mediciones"; ?></td>
            <td style="width: 120px; text-align: center" class="bordesAmarillos""> 
<?php echo isset($modeloLast) ? (isset($model->VibLA)) ? $modeloLast->VibLA : "" : "No se encontraron mediciones"; ?></td>
            <td style="width: 80px; text-align: center;" class="bordesAmarillos">
<?php echo isset($modeloLast) ? (isset($model->Temperatura)) ? $modeloLast->Temperatura : "" : "No se encontraron mediciones"; ?></td>
        </tr>
        <tr>
            <td style="width: 80px; text-align: center" class="bordesAmarillos">
                <?php echo CHtml::image($imagenVMotor); ?>

            </td>
            <td style="width: 120px; text-align: center" class="bordesAmarillos"">
<?php echo CHtml::image($imagenVBomba); ?>
        </td>
        <td style="width: 80px; text-align: center" class="bordesAmarillos">
        </td>
    </tr>
</table>

</div>                



            


<div style="width:78%;float:left; padding-top:8px;">





    <?php

// arreglos para gaficar
    $arrSalidaTemperatura = array();
// obtiene una array de modelos que tienen el tag= al del modelo actual
    $data = Vibraciones::model()->findAllBySQL('SELECT * FROM VIBRACIONES WHERE TAG="'.$TAG.'" ORDER BY FECHA ASC');
// para cada uno de los modelos calcula el IP de cada fase y lo almacena en formato de stockhigh
    $contador = 0;
    foreach ($data as $modeloTemp) {
        // hace push en arrSalidaA,B,C de un array(strtotime($foreignobj->Fecha)*1000,$foreignobj->A10/$foreignobj->A1))
        array_push($arrSalidaTemperatura, array(strtotime($modeloTemp->Fecha) * 1000, 0 + $modeloTemp->Temperatura));
        $contador = $contador + 1;
    }
    if ($contador == 0) {
        echo "No hay datos para graficar.";
    } else {
//print_r($arrSalidaTemperatura);
// hace el gráfico de IP
        $this->Widget('ext.highstock.HighstockWidget', array(
            'id' => 'grafico2',
            'options' => array(
                'colors' => array('#07DD07'),
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
                        'fontSize' => '8px'
                    )
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
                'colors' => array('green', 'red', 'blue'),
                'credits' => array('enabled' => false),
                'legend' => array('enabled' => false),
                'theme' => 'grid',
                'title' => array('text' => 'Tendencia Temperatura más Alta Conjunto Motriz'),
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
                ))
        );
        ?>

        <div class="forms50c forms50cl">
        <?php
// guarda en $TAG_in el TAG del modelo actual

        echo "<br/>";
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
                'colors' => array('blue', 'red', 'green'),
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
                        'fontSize' => '8px'
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
<div class="forms50c forms50cr">
<?php
// guarda en $TAG_in el TAG del modelo actual

                    echo "<br/>";
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
                            'colors' => array('#F42539'),
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
                                    'fontSize' => '8px'
                                ),
                            ),
                            'title' => array('text' => 'Vibr. L. Bomba o Reductor'),
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
    <?php
    }
}
?>

    </div>    
        <div style="width:20%;float:right;">

<?php
$this->widget('zii.widgets.grid.CGridView', array(
    'id' => 'aislamiento-tierra-grid',
    'dataProvider' => Vibraciones::model()->searchFechas($TAG),
    //  'filter' => $model,
    'cssFile' => '/themes/gridview/styles.css', 'template' => '{items}{pager}{summary}', 'summaryText' => 'Resultados del {start} al {end} de {count} encontrados',
    'columns' => array(
        //	'id',
        //	'Toma',
        //   'TAG',
        //   'relacMotores.Area',
        'Fecha', array(
            'class' => 'CButtonColumn',
            'template' => $dependeT,
        ),
    ),
));
?>
</div>
