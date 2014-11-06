
<?php
$this->breadcrumbs = array(
    'Informes de Tableros Eléctricos' => array('index'),
    $model->id,
);

// busca el modelo del equipo en Tableros
$modelTMP = Tableros::model()->findByAttributes(array('TAG' => $model->TAG));
$suffixID="";
if (isset($modelTMP))
{
    $suffixID=$modelTMP->id;
}
$this->menu = array(
    array('label' => 'Lista de Informes', 'url' => array('index')),
    array('label' => 'Nuevo Informe', 'url' => array('/termotablero/create?id=' . $model->TAG)),
    array('label' => 'Gestionar Informes', 'url' => array('/termotablero/admin?id=' .$suffixID)),
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
    array_push($this->menu, array('label' => 'Actualizar Informe', 'url' => array('update', 'id' => $model->id)));
if ($esAdmin)
    array_push($this->menu, array('label' => 'Borrar Informe', 'url' => '#', 'linkOptions' => array('submit' => array('delete', 'id' => $model->id), 'confirm' => 'EstÃ¡ seguro de borrar esto?')));


?>

<?php
$nombre = "";
$modelTMP = $model;
if (isset($modelTMP->TAG))
    $nombre = $modelTMP->TAG;
if (isset($modelTMP->Fecha))
    $nombre = $nombre . ' (' . $modelTMP->Fecha . ')';
?>
<?php $this->setPageTitle('Detalles de informe de termografia de:<br/>'.$nombre.''); ?>

<?php
$modelT = Tableros::model()->findByAttributes(array('TAG' => $model->TAG));
$modelM = Tableros::model()->findByAttributes(array('TAG' => $model->TAG));
?>
<?php
    function colorEstado($EstadoIn)
    {
        if ($EstadoIn==0) return('<img src="/images/verde.gif" height="15" width="15" /> 0 - Adecuado');
        if ($EstadoIn==1) return('<img src="/images/amarillo.gif" height="15" width="15" /> 1 - Posible deficiencia - Se requiere más información.');
        if ($EstadoIn==2) return('<img src="/images/amarillo.gif" height="15" width="15" /> 2 - Posible deficiencia - Reparar en la próxima parada disponible');
        if ($EstadoIn==3) return('<img src="/images/rojo.gif" height="15" width="15" /> 3 - Deficiencia - Reparar tán pronto como sea posible');
        if ($EstadoIn==4) return('<img src="/images/rojo.gif" height="15" width="15" /> 4 - Deficiencia Grave - Inmediatamente');
        
    }
?>
<div  class="forms100c">
<?php


$this->widget('zii.widgets.CDetailView', array(
    'data' => $model,
    'cssFile' => '/themes/detailview/styles.css',
    'attributes' => array(
        //'id',
        //'Fecha',
        
        array(// related city displayed as a link
            'label' => 'Tablero',
            'type' => 'raw',
            'value' => ((isset($model->TAG)) ? $model->TAG : "") . " - " . ((isset($modelT->Tablero)) ? $modelT->Tablero : ""),
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
        'OT',
                array(
                    'label'=>'Estado',
                    'value'=> colorEstado($model->Estado),
                    'htmlOptions'=>array('style'=>'width:20px;'),
                    'type'=>'raw',),
        
        //'Criterio',
        //'Tamano',
        'Analista',
        'Observaciones',
        //'Path',
        array(// related city displayed as a link
            'label' => 'Path',
            'type' => 'raw',
            'value' => (isset($model->Path)) ? (is_numeric($model->Path)?  '<a href="/index.php/archivos/displayArchivo?id='.$model->Path.'">Descargar Informe</a>' :'<a href="/index.php/reportes/passthru?path='.urlencode($model->Path).'">'.$model->Path.'</a>'): "",
        ),

    ),
    
));
?>
</div>


