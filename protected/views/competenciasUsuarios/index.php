<?php
$this->breadcrumbs = array(
	'Competencias Usuarioses',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Create') . ' CompetenciasUsuarios', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Manage') . ' CompetenciasUsuarios', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Competencias Usuarioses'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
