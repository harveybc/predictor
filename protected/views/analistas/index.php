<?php
$this->breadcrumbs = array(
	'Analistas',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Crear') . ' Analista', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Analistas', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Analistas'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
