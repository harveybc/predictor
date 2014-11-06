<?php
$this->breadcrumbs = array(
	'Préstamos',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Crear') . ' Préstamos', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Préstamos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Préstamos'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
