<?php
$this->breadcrumbs = array(
	'Idiomas',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Crear') . ' Idiomas', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Idiomas', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Idiomas'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
