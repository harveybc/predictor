

<?php
$this->breadcrumbs = array(
    'Lubricantes' => array('index'),
    $model->id,
);
// busca el modelo del equipo en Motores
$modelTMP = Motores::model()->findByAttributes(array('TAG' => $model->TAG));
$suffixID = "";
if (isset($modelTMP)) {
    $suffixID = $modelTMP->id;
}
$this->menu = array(
    array('label' => Yii::t('app', 'Instrucciones'), 'url' => array('/Archivos/displayArchivo?id=29')),
    array('label' => 'Lista de Mediciones', 'url' => array('index')),
    array('label' => 'Crear Medición', 'url' => array('/aceitesnivel1/create?id=' . $model->TAG)),
    array('label' => 'Gestionar Mediciones', 'url' => array('/aceitesnivel1/admin?id=' . $suffixID))
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
$nombre = "";
$modelTMP = $model;
if (isset($modelTMP->TAG))
    $nombre = $modelTMP->TAG;
if (isset($modelTMP->Fecha))
    $nombre = $nombre . ' (' . $modelTMP->Fecha . ')';
?>
<?php $this->setPageTitle(' Detalles de medición de Lubricantes:<br/>'.$nombre.''); ?>

<?php
$modelN1 = Motores::model()->findByAttributes(array('TAG' => $model->TAG));
?>

<?php
$this->widget('zii.widgets.CDetailView', array(
    'data' => $model,
    'cssFile' => '/themes/detailview/styles.css',
    'attributes' => array(
        //'id',
        //'Toma',
        //'TAG',
        array(// related city displayed as a link
            'label' => 'Motor',
            'type' => 'raw',
            'value' => ((isset($model->TAG)) ? $model->TAG : "") . " - " . ((isset($modelN1->Motor)) ? $modelN1->Motor : ""),
        ),
        array(// related city displayed as a link
            'label' => 'Equipo',
            'type' => 'raw',
            'value' => (isset($modelN1->Equipo)) ? $modelN1->Equipo : "",
        ),
        array(// related city displayed as a link
            'label' => 'Proceso',
            'type' => 'raw',
            'value' => (isset($modelN1->Area)) ? $modelN1->Area : "",
        ),
        array(// related city displayed as a link
            'label' => 'Area',
            'type' => 'raw',
            'value' => (isset($modelN1->Proceso)) ? $modelN1->Proceso : "",
        ),
        //'Fecha',
        'OT',
        'Analista',
        'Medicion',
        'Tipo',
    ),
));
?>
<style>
    .bordesAmarillos{
        width:280px;padding:0px;margin:0px;border-color:#FCF4A1;border-width: 1px;border-style:solid;text-align:right;
    }
</style>

<table class="bordesAmarillos">
    <tr>
        <td style="width:300px;text-align:right">
            <b>Estado de Aceite</b>
        </td>
<?php
if ($model->Estado == 2)
    echo '<td style="background-color:red;width:250px;font-color:#ffffff;text-align:center;">Malo</td>';
if ($model->Estado == 1)
    echo '<td style="background-color:orange;width:250px;font-color:#ffffff;text-align:center;">Medio</td>';
if ($model->Estado == 0)
    echo '<td style="background-color:#00CC00;width:250px;font-color:#ffffff;text-align:center;">Bueno</td>';
?>
    </tr>

</table>
<br/>
<div style="width:100%">
<?php
// arreglos para gaficar
$arrSalidaA = array();

// obtiene una array de modelos que tienen el tag
$data = Aceitesnivel1::model()->findAllBySql("select * from aceitesnivel1 where TAG=\"" . $model->TAG . "\" order by Fecha ASC");
// para cada uno de los modelos calcula el IP de cada fase y lo almacena en formato de stockhigh
foreach ($data as $foreignobj) {
    // hace push en arrSalidaA,B,C de un array(strtotime($foreignobj->Fecha)*1000,$foreignobj->A10/$foreignobj->A1))
    //calcula los valores para graficar el estado
    if (isset($foreignobj->Estado)) {
        $grVal = 0;
        if ($foreignobj->Estado == 0)
            $grVal = 2;
        if ($foreignobj->Estado == 1)
            $grVal = 1;
        array_push($arrSalidaA, array(strtotime($foreignobj->Fecha) * 1000, $grVal));
    }
}
// hace el grÃ¡fico de IP
$this->Widget('ext.highstock.HighstockWidget', array(
    'id' => 'grafico2',
    'options' => array(
        'colors' => array('6caddf',),
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
        'credits' => array('enabled' => false),
        'legend' => array('enabled' => false),
        'theme' => 'grid',
        'rangeSelector' => array('selected' => 6
        ),
        'title' => array('text' => 'Estado de Aceite'),
        'xAxis' => array('maxZoom' => 14 * 24 * 360000),
        'yAxis' => array('title' => array('text' => 'Estado')),
        'series' => array(
            array('name' => 'Estado', 'data' => $arrSalidaA),
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