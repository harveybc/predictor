<?php
$this->breadcrumbs = array(
	'Aislamiento Fase',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Crear') . ' Mediciones', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Mediciones', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Aislamiento Fases'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
