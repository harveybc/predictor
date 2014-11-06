<?php
$this->breadcrumbs = array(
	'Reportes',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Crear') . ' Reporte', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Reporte', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Reportes de fugas de gases (Ultrasonido)'); ?>
<div class="forms100c" style="text-align:left;">
<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
</div>