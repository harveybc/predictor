<?php
$this->breadcrumbs = array(
    'Tableros' => array('index'),
    $model->id,
);

// busca el modelo del equipo en Tableros
$modelTAB = Tableros::model()->findByAttributes(array('Area' => $model->Area));

$this->menu = array(
    array('label' => 'Lista de Tableros', 'url' => array('index')),
    array('label' => 'Gestionar Tableros', 'url' => array('/tableros/admin?proceso=' . $modelTAB->Area)),
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
{
    array_push($this->menu, array('label'=>'Nuevo Tablero', 'url'=>array('/tableros/create?id=' . $model->Area)));
    array_push($this->menu,array('label'=>'Actualizar Tablero', 'url'=>array('update', 'id'=>$model->id)));
}
if ($esAdmin) array_push($this->menu,array('label'=>'Borrar Tablero', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->id),'confirm'=>'Está seguro de borrar esto?')));


?>

<?php
$nombre = "";
$modelTMP = Tableros::model()->findByAttributes(array('TAG' => $model->TAG));
if (isset($modelTMP->Tablero))
    $nombre = $modelTMP->Tablero;
if (isset($modelTMP->TAG))
    $nombre = $nombre . ' (' . $modelTMP->TAG . ')';
?>

<?php $this->setPageTitle ('Detalles Tablero:'.$nombre.''); ?>

<?php
$model1 = Tableros::model()->findByAttributes(array('TAG' => $model->TAG));
?>

<?php
$this->widget('zii.widgets.CDetailView', array(
    'data' => $model,
    'cssFile' => '/themes/detailview/styles.css',
    'attributes' => array(
        //'id',
        array(// related city displayed as a link
            'label' => 'Tablero',
            'type' => 'raw',
            'value' => ((isset($model->TAG)) ? $model->TAG : "") . " - " . ((isset($model1->Tablero)) ? $model1->Tablero : ""),
        ),
        'Area',
        'Proceso',
        //'plan_mant_termografia'
    //'TAG',
    //'Tablero',
    ),
));
?>


