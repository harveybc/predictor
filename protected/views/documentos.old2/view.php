<?php
$this->breadcrumbs = array(
    'Unidades Documentales' => array('index'),
    $model->id,
);

$this->menu = array(
    array('label' => 'Lista de Unidades', 'url' => array('index')),
    array('label' => 'Nueva Unidad', 'url' => array('create')),
    array('label' => 'Actualizar Unidad', 'url' => array('update', 'id' => $model->id)),
    array('label' => 'Borrar Unidad', 'url' => '#', 'linkOptions' => array('submit' => array('delete', 'id' => $model->id), 'confirm' => 'Está seguro de borrar esto?')),
    array('label' => 'Gestionar Unidades', 'url' => array('admin')),
);
?>

<?php $this->setPageTitle('Detalles de Unidad Documental:<?php echo $model->descripcion; ?>'); ?>

<?php
$this->widget('zii.widgets.CDetailView', array(
    'data' => $model,
    'cssFile' => '/themes/detailview/styles.css',
    'attributes' => array(
        //'id',
        'descripcion',
        'permitirAdiciones',
        'permitirAnotaciones',
        'autorizarOtros',
        'requiereAutorizacion',
        'secuencia0.descripcion',
        'ordenSecuencia0.posicion',
        'eliminado',
        'conservacionInicio',
        'conservacionFin',
        'conservacionPermanente',
    ),
));
?>
<br/>
<h2 style="font-size:18px;color:#961C1F;"> Esta unidad documental contiene:'); ?>

<h2  style="font-size:16px;;"> Documentos: '); ?>
<ul><?php
foreach ($model->metaDocs as $foreignobj) {
    // Para mostrar los documentos electrónicos (Faltan los físicos y los online)
    if (isset($foreignobj->ruta))
        printf('<li>%s</li>', CHtml::link($foreignobj->titulo, '/index.php/MetaDocs/view?id='.$foreignobj->id));
}
?></ul><br />
