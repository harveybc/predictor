<?php
$this->breadcrumbs = array(
	'Aislamiento Tierra',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Crear') . ' Medición Aislamiento tierra', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . 'Medición Aislamiento tierra', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Aislamiento Tierra'); ?>
<div class="forms100c" style="text-align:left;">
<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
</div>