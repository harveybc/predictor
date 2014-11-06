<style>
    td
    {
        border: 1px solid #AC8B3A;
        text-align:center;
        width: 50px;
        font-size: 12px;
        margin:0px;
        padding:0px;
    }
    td.sinborde
    {
        border-bottom: 1px solid #AC8B3A;
        border-right: 1px solid #AC8B3A;
        border-top: 0px;
        border-left: 0px;
        text-align:center;
        width: 50px;
        font-size: 10px;
        margin:0px;
        padding:0px;
    }
    .marcoDoc{
        width:100%;border-color:#AC8B3A;
        
        -moz-border-radius: 3px; /* Firefox */
        -webkit-border-radius: 3px; /* Safari and Chrome */
        -border-radius: 3px;
    }
</style>
<?php
    function colorEstado($modeloIn)
    {
        $cadSalida1="No hay datos para este motor";
        $modelTMP = $modeloIn;
        if (isset($modelTMP))
        {
            $EstadoIn=$modelTMP->Estado;
            if ($EstadoIn==0) $cadSalida1='<img src="/images/verde.gif" height="15" width="15" /> Bueno';
            if ($EstadoIn==1) $cadSalida1='<img src="/images/amarillo.gif" height="15" width="15" /> Atención';
            if ($EstadoIn==2) $cadSalida1='<img src="/images/rojo.gif" height="15" width="15" /> 2 - Malo';
            if ($EstadoIn==3) $cadSalida1='<img src="/images/rojo.gif" height="15" width="15" /> 3 - Malo';
        }
        return($cadSalida1);
    }
?>
<?php
$this->breadcrumbs = array(
    'Aislamiento Tierra' => array('index'),
    $model->Toma,
);


// busca el modelo del equipo en Motores
$modelTMP1 = Motores::model()->findByAttributes(array('TAG' => $model->TAG));

$suffixID="";
if (isset($modelTMP1))
{
    $suffixID=$modelTMP1->id;
}

$this->menu = array(
    array('label' => Yii::t('app', 'Instrucciones'), 'url' => array('/Archivos/displayArchivo?id=27')),
    array('label' => 'Lista de mediciones', 'url' => array('index')),
    array('label' => 'Nueva medida', 'url' => array('/aislamiento_tierra/create?id=' . $model->TAG)),
    array('label' => 'Gestionar medición', 'url' => array('/aislamiento_tierra/admin?id=' . $suffixID))
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
    array_push($this->menu, array('label' => 'Actualizar Medición', 'url' => array('update', 'id' => $model->Toma)));
if ($esAdmin)
    array_push($this->menu, array('label' => 'Borrar Medición', 'url' => '#', 'linkOptions' => array('submit' => array('delete', 'id' => $model->Toma), 'confirm' => 'EstÃ¡ seguro de borrar esto?')));


?>
<?php
$nombre = "";
$sufix = "#";
$modelTMP = Motores::model()->findByAttributes(array('TAG' => $model->TAG));
if (isset($modelTMP))
{

if (isset($modelTMP->Motor))
    $nombre = $modelTMP->Motor;
if (isset($modelTMP->TAG))
{
    $nombre = $nombre . ' (' . $modelTMP->TAG . ')';
    $sufix = "/index.php/site/buscar?query=" . urlencode($modelTMP->TAG);
    
}
else
    $sufix = "#";
}
?>
<?php 
echo "<a href=".$sufix.">";

?>
<?php $this->setPageTitle('Detalles de medición  de Aislamiento Tierra<br/> 
<?php 
echo $nombre; 
?>
'); ?>



<?php
    echo "</a>";
    $modelAT = Motores::model()->findByAttributes(array('TAG' => $model->TAG));
?>
Estado: <?php echo colorEstado($model)?>

<br/>
<table class="marcoDoc">
    <tr style="height:10px;">
        
         <td>
            <b>TAG</b>
        </td>
        <td>
            <b>Motor</b>
        </td>
        <td>
            <b>Equipo</b>
        </td>
        <td>
            <b>Proceso</b>
        </td>
        <td>
            <b>Area</b>
        </td>
        <td>
            <b>O.T</b>
        </td>

    </tr>
    <tr>
        <td style="width:30px">
<?php echo $model->TAG ?>

        </td>
        
          <td style="width:200px">
<?php echo $modelAT->Motor; ?>
        </td>
        <td style="width:100px">
<?php echo $modelAT->Equipo; ?>
        </td>
        <td style="width:100px">
            <?php echo $modelAT->Area; ?>
        </td>
        <td style="width:100px">
            <?php echo $modelAT->Proceso; ?>
        </td>
        <td style="width:100px">
<?php echo $model->OT; ?>
        </td>


    </tr>

</table>
<legend><b>Fase / Min.</b></legend>

<table>

    <tr style="color:#ff0000;">
        <td class="sinborde">
        </td>
        <td>
            <b>  IP </b>
        </td>
        <td>
            <b>  0.25 </b>
        </td>
        <td>
            <b>  0.5 </b>
        </td>
        <td>
            <b> 1  </b>
        </td>
        <td>
            <b>  2  </b>
        </td>
        <td>
            <b>  3  </b>
        </td>
        <td>
            <b>  4  </b>
        </td>
        <td>
            <b> 5  </b>
        </td>
        <td>
            <b>6  </b>
        </td>
        <td>
            <b>  7  </b>
        </td>
        <td>
            <b>  8  </b>
        </td>
        <td>
            <b>   9  </b>
        </td>
        <td>
            <b>  10  </b>
        </td>
    </tr>
    <tr>
        <td style="color:#961C1F;">
            <b>    A    </b>    
        </td>
        <td>
            <?php echo ($model->A1!=0)?($model->A10/$model->A1):"División por CERO"; ?>
        </td>
        <td>
<?php echo $model->A025; ?>
        </td>
        <td>
<?php echo $model->A050; ?>
        </td>
        <td>
            <?php echo $model->A1; ?>
        </td>
        <td>
            <?php echo $model->A2; ?>
        </td>
        <td>
            <?php echo $model->A3; ?>
        </td>
        <td>
            <?php echo $model->A4; ?>
        </td>
        <td>
            <?php echo $model->A5; ?>
        </td>

        <td>
            <?php echo $model->A6; ?>
        </td>

        <td>
<?php echo $model->A7; ?>
        </td>
        <td>
            <?php echo $model->A8; ?>
            </div

        </td>
        <td>
<?php echo $model->A9; ?>
        </td>
        <td>
<?php echo $model->A10; ?>
        </td>
    </tr>
    <tr>
        <td style="color:#961C1F;color:#961C1F">
            <b>   B </b>
        </td>
        <td>
            <?php echo ($model->B1!=0)?($model->B10/$model->B1):"División por CERO"; ?>
        </td>
        <td>
<?php echo $model->B025; ?>
        </td>
        <td>
<?php echo $model->B050; ?>
        </td>
        <td>
            <?php echo $model->B1; ?>
        </td>
        <td>
            <?php echo $model->B2; ?>
        </td>
        <td>
            <?php echo $model->B3; ?>
        </td>
        <td>
            <?php echo $model->B4; ?>
        </td>
        <td>
            <?php echo $model->B5; ?>
        </td>
        <td>
            <?php echo $model->B6; ?>
        </td>
        <td>
            <?php echo $model->B7; ?>
        </td>
        <td>
            <?php echo $model->B8; ?>
        </td>
        <td>
            <?php echo $model->B9; ?>
        </td>
        <td>
            <?php echo $model->B10; ?>
        </td>
    </tr>
    <tr>
        <td style="color:#961C1F;">
            <b>  C </b>
        </td>
        <td>
            <?php echo ($model->C1!=0)?($model->C10/$model->C1):"División por CERO"; ?>
        </td>
        <td>
<?php echo $model->C025; ?>
        </td>
        <td>
<?php echo $model->C050; ?>
        </td>
        <td>
            <?php echo $model->C1; ?>
        </td>
        <td>
            <?php echo $model->C2; ?>
        </td>
        <td>
            <?php echo $model->C3; ?>
        </td>
        <td>
            <?php echo $model->C4; ?>
        </td>
        <td>
            <?php echo $model->C5; ?>
        </td>
        <td>
            <?php echo $model->C6; ?>
        </td>
        <td>
            <?php echo $model->C7; ?>
        </td>
        <td>
            <?php echo $model->C8; ?>
        </td>
        <td>
            <?php echo $model->C9; ?>
        </td>
        <td>
            <?php echo $model->C10; ?>
        </td>
    </tr>


</table>

<br/>
<table>
    <tr>
        <td style='border: 0px;width:50%'>
            <div id="divGraphFases">
<?php
// Consulta si existe un registro para la toma actual en aislamiento_fases
$tmpModel = Aislamiento_fases::model()->findByAttributes(array('Toma' => $model->Toma));
// Si NO existe, muestra Link para crear nuevo con $_GET=Toma actual
if ($tmpModel == NULL)
    echo CHtml::link('Nueva Medición de Aislamiento Fases', array('/aislamiento_fases/create', 'Toma' => $model->Toma));
// Si existe,
else {
    // Muestra tabla con Datos

    echo "
                            <table>
                             <legend><b> Medición de Aislamiento (Mohms) Fase a Fase</b> </legend>
                                <tr >
                                    <td>
                                    
                                    <b>Fases/Min.<br/>
                                    </td>
                                    <td>
                                  <b>  0.25 </b>
                                    </td>
                                    <td>
                                  <b>  0.5 </b>
                                    </td>
                                    <td>
                                   <b> 1 </b>
                                    </td>
                                    <td>
                                   <b> 2 </b>
                                    </td>
                                    <td>
                                <b>    Indice de<br/>
                                    Absorción
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                     <b>A-B </b>
                                    </td>
                                    <td>
                                        " . $tmpModel->A025 . "  
                                    </td>                                    
                                    <td>
                                        " . $tmpModel->A050 . "  
                                    </td>
                                    <td>
                                       " . $tmpModel->A1 . " 
                                    </td>
                                    <td>
                                       " . $tmpModel->A2 . " 
                                    </td>
                                    <td>
                                        " . $tmpModel->A1 / $tmpModel->A050 . "
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                  <b>  B-C </b>
                                    </td>
                                    <td>
                                        " . $tmpModel->B025 . "
                                    </td>
                                    <td>
                                        " . $tmpModel->B050 . "
                                    </td>
                                    <td>
                                        " . $tmpModel->B1 . "
                                    </td>
                                    <td>
                                        " . $tmpModel->B2 . "
                                    </td>
                                    <td>
                                        " . $tmpModel->B1 / $tmpModel->B050 . "
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                     <b> C-A </b>
                                    </td>
                                    <td>
                                        " . $tmpModel->C025 . "
                                    </td>
                                    <td>
                                        " . $tmpModel->C050 . "
                                    </td>
                                    <td>
                                        " . $tmpModel->C1 . "
                                    </td>
                                    <td>
                                        " . $tmpModel->C2 . "
                                    </td>
                                    <td>
                                        " . $tmpModel->C1 / $tmpModel->C050 . "
                                    </td>
                                </tr>
                            </table>
                            <br/>
                            ";
    // Muestra links para editar
    echo CHtml::link('Editar', array('/aislamiento_fases/update/' . $tmpModel->Toma));
}
?>
            </div>
        </td>
        <td style="border: 0px;padding-left:8px;width:50%">
            <div id="divGraphAcometida">
                <?php
                // Consulta si existe un registro para la toma actual en aislamiento_fases
                $tmpModel = Aislamiento_acometida::model()->findByAttributes(array('Toma' => $model->Toma));
                // Si NO existe, muestra Link para crear nuevo con $_GET=Toma actual
                if ($tmpModel == NULL)
                    echo CHtml::link('Nueva Medición de Aislamiento Acometida', array('/aislamiento_acometida/create', 'Toma' => $model->Toma));
                // Si existe,
                else {
                    // Muestra tabla con Datos
                    if (($tmpModel->A050!=0)&&($tmpModel->B050!=0)&&($tmpModel->A050!=0))
                    echo "
                            <table>
                            <legend><b> Medición de Aislamiento (Mohms) Acometida</b> </legend>
                                <tr>
                                    <td>
                                    <b>Fases/Min.<br/>
                                    </td>
                                    <td>
                                  <b>  0.25 </b>
                                    </td>
                                    <td>
                                  <b>  0.5 </b>
                                    </td>
                                    <td>
                                   <b> 1 </b>
                                    </td>
                                    <td>
                                   <b> 2 </b>
                                    </td>
                                    <td>
                               <b>     Indice de<br/>
                                    Absorción
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                    <b> A-B </b>
                                    </td>
                                    <td>
                                        " . $tmpModel->A025 . "
                                    </td>
                                    <td>
                                        " . $tmpModel->A050 . "
                                    </td>
                                    <td>
                                        " . $tmpModel->A1 . "
                                    </td>
                                    <td>
                                        " . $tmpModel->A2 . "
                                    </td>
                                    <td>
                                        " . $tmpModel->A1 / $tmpModel->A050 . "
                                    </td>
                                </tr>
                                <tr>
                                   <td>
                                   <b> B-C </b>
                                    </td>
                                    <td>
                                        " . $tmpModel->B025 . "
                                    </td>
                                    <td>
                                        " . $tmpModel->B050 . "
                                    </td>
                                    <td>
                                        " . $tmpModel->B1 . "
                                    </td>
                                    <td>
                                        " . $tmpModel->B2 . "
                                    </td>
                                    <td>
                                        " . $tmpModel->B1 / $tmpModel->B050 . "
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                    <b> C-A </b>
                                    </td>
                                    <td>
                                        " . $tmpModel->C025 . "
                                    </td>                                    
                                    <td>
                                        " . $tmpModel->C050 . "
                                    </td>
                                    <td>
                                        " . $tmpModel->C1 . "
                                    </td>
                                    <td>
                                        " . $tmpModel->C2 . "
                                    </td>
                                    <td>
                                        " . $tmpModel->C1 / $tmpModel->C050 . "
                                    </td>
                                </tr>
                            </table>
                            <br/>
                            ";
                    // Muestra links para editar
                    echo CHtml::link('Editar', array('/aislamiento_acometida/update/' . $tmpModel->Toma));
                }
                ?>
            </div>
        </td>
    </tr>    
</table>


<div class="forms50c forms50cl">
                <?php
                $TAG_in = $model->TAG;
                if ($TAG_in == "") {
                    echo "No hay datos para graficar.";
                    return;
                }
// arreglos para gaficar
                $arrSalidaA = array(
                    0 + $model->A025, 0 + $model->A050, 0 + $model->A1, 0 + $model->A2, 0 + $model->A3, 0 + $model->A4, 0 + $model->A5, 0 + $model->A6, 0 + $model->A7, 0 + $model->A8, 0 + $model->A9, 0 + $model->A10
                );
                $arrSalidaB = array(
                    0 + $model->B025, 0 + $model->B050, 0 + $model->B1, 0 + $model->B2, 0 + $model->B3, 0 + $model->B4, 0 + $model->B5, 0 + $model->B6, 0 + $model->B7, 0 + $model->B8, 0 + $model->B9, 0 + $model->B10
                );
                $arrSalidaC = array(
                    0 + $model->C025, 0 + $model->C050, 0 + $model->C1, 0 + $model->C2, 0 + $model->C3, 0 + $model->C4, 0 + $model->C5, 0 + $model->C6, 0 + $model->C7, 0 + $model->C8, 0 + $model->C9, 0 + $model->C10
                );
                $arrLegend = array("0.25", "0.5", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10");
// Hace la gráfica de La medición (12 valores) para cada fase
                $this->Widget('ext.highcharts.HighchartsWidget', array(
                    'id' => 'grafico1',
                    'options' => array(
                        'colors' => array('#6caddf', '#8bc63e', '#f79727'),
                        'chart' => array(
                            'selectionMarkerFill' => '#F6EAA5',
                            'backgroundColor' => '#FCFAE8',
                            'borderColor' => '#e0cd7c',
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
                        //'legend' => array('enabled' => true),
                        'title' => array('text' => 'Curva Aislamiento Fase - Tierra'),
                        'xAxis' => array(
                            'title' => array('text' => 'Min.'),
                            'categories' => $arrLegend,
                        //'categories' => array(4, 5, 6)
                        ),
                        'theme' => 'default',
                        'yAxis' => array(
                            'title' => array('text' => 'Valor')
                        ),
                        'credits' => array('enabled' => false),
                        'legend' => array('enabled' => false),
                        'series' => array(
                            array('name' => 'A', 'data' => $arrSalidaA),
                            array('name' => 'B', 'data' => $arrSalidaB),
                            array('name' => 'C', 'data' => $arrSalidaC)
                        //array('name' => 'Valor', 'data' => array(1, 2, 3))
                        ),
                    )
                ));
                echo '</div>';
echo '<div class="forms50c forms50cr">';                

                
// arreglos para gaficar
                $arrSalidaA = array();
                $arrSalidaB = array();
                $arrSalidaC = array();
// obtiene una array de modelos que tienen el tag
                $data = Aislamiento_tierra::model()->findAllBySql("select * from aislamiento_tierra where TAG=\"" . $TAG_in . "\" order by Fecha ASC");
// para cada uno de los modelos calcula el IP de cada fase y lo almacena en formato de stockhigh
                foreach ($data as $foreignobj) {
                    // hace push en arrSalidaA,B,C de un array(strtotime($foreignobj->Fecha)*1000,$foreignobj->A10/$foreignobj->A1))
                    if (isset($foreignobj->A1))
                        if (0 + $foreignobj->A1 > 0)
                            array_push($arrSalidaA, array(strtotime($foreignobj->Fecha) * 1000, $foreignobj->A10 / $foreignobj->A1));
                    if (isset($foreignobj->B1))
                        if (0 + $foreignobj->B1 > 0)
                            array_push($arrSalidaB, array(strtotime($foreignobj->Fecha) * 1000, $foreignobj->B10 / $foreignobj->B1));
                    if (isset($foreignobj->C1))
                        if (0 + $foreignobj->C1 > 0)
                            array_push($arrSalidaC, array(strtotime($foreignobj->Fecha) * 1000, $foreignobj->C10 / $foreignobj->C1));
                }
// hace el gráfico de IP
                $this->Widget('ext.highstock.HighstockWidget', array(
                    'id' => 'grafico2',
                    'options' => array(
                        'colors' => array('#6caddf', '#8bc63e', '#f79727'),
                        'chart' => array(
                            'selectionMarkerFill' => '#F6EAA5',
                            'backgroundColor' => '#FCFAE8',
                            'borderColor' => '#e0cd7c',
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
                        'credits' => array('enabled' => false),
                        'legend' => array('enabled' => false),
                        'theme' => 'grid',
                        'rangeSelector' => array('selected' => 6
                        ),
                        'title' => array('text' => 'Indice de Potencia de Fases A, B, C'),
                        'xAxis' => array('maxZoom' => 14 * 24 * 360000),
                        'yAxis' => array('title' => array('text' => 'Indice de Potencia')),
                        'series' => array(
                            array('name' => 'Fase A', 'data' => $arrSalidaA),
                            array('name' => 'Fase B', 'data' => $arrSalidaB),
                            array('name' => 'Fase C', 'data' => $arrSalidaC),
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