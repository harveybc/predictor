<?php
$this->breadcrumbs = array(
	'Aislamiento Acometida',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Crear') . ' Medición', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Medición', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Aislamiento Acometida'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
