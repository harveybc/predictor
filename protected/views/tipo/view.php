<?php
$this->breadcrumbs = array(
    'Lubricantes' => array('index'),
    $model->id,
);

$this->menu = array(
    array('label' => 'Lista de Lubricantes', 'url' => array('index')),
    array('label' => 'Nuevo Lubricante', 'url' => array('/tipo/create?id=' . $model->Proceso)),
    array('label' => 'Actualizar Lubricante', 'url' => array('update', 'id' => $model->id)),
    array('label' => 'Borrar Lubricante', 'url' => '#', 'linkOptions' => array('submit' => array('delete', 'id' => $model->id), 'confirm' => 'Está seguro de borrar esto?')),
    array('label' => 'Gestionar Lubricantes', 'url' => array('admin')),
);
?>

<?php
$nombre = "";
$modelTMP = $model;
if (isset($modelTMP->Tipo_Aceite))
    $nombre = $modelTMP->Tipo_Aceite;
?>
<?php $this->setPageTitle('Detalles Lubricante:'.$nombre.''); ?>

<?php
$this->widget('zii.widgets.CDetailView', array(
    'data' => $model,
    'cssFile' => '/themes/detailview/styles.css',
    'attributes' => array(
        //'id',
        'Proceso',
        'Tipo_Aceite',
    ),
));
?>


